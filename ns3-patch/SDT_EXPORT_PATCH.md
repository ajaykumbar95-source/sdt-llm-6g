# Patch: export per-path multipath data from SionnaRtChannelModel

Adds two small pieces to `src/spectrum/model/sionna-rt-channel-model.{h,cc}`
so that every time ns-3 resolves the channel via Sionna RT, it also appends
one CSV row per path to a file that `sdt_llm/data/ns3_stream_bridge.py` (on
the Python/SDT side) tails live. Nothing else in the class changes — this
is purely additive.

Written against the exact headers you shared (thank you — very thorough
docstrings, which is most of why this patch could be written precisely
instead of guessed). NOT compile-tested (no ns-3 build in the environment
this was written in) — see "If it doesn't compile" at the bottom.

--------------------------------------------------------------------------
Why bistatic, not monostatic
--------------------------------------------------------------------------
`GetNewChannel`'s own docs say `sMob` = transmitter, `uMob` = receiver —
i.e. a real gNB<->UE NR link, generally at DIFFERENT positions. So this
patch writes BOTH positions, and the Python side now does exact bistatic
geometry (solves for the scatterer on the tx/rx ellipse, not a naive
round-trip-from-one-point range) — see `bistatic_scatterer_location()` in
`synthetic_radio.py` and `tests/test_bistatic_localization.py`. It reduces
exactly to the simple monostatic formula when tx and rx coincide, so this
is correct either way, not a tradeoff.

==========================================================================
STEP 1 — src/spectrum/model/sionna-rt-channel-model.h
==========================================================================

Find this (the last method in the public section, right before `protected:`):

    RtPathSolverConfig GetRtPathSolverConfig() const;

  protected:

Change it to (adding one new public method before `protected:`):

    RtPathSolverConfig GetRtPathSolverConfig() const;

    /**
     * @brief Enable exporting per-path multipath data (delay, angles,
     * Doppler, complex gain) to a CSV file, for an external pipeline (e.g.
     * a semantic digital twin) to consume live. Disabled by default -- no
     * export happens until this is called. Call once, e.g. at the top of
     * main(), before running the simulation. Rows are appended, not
     * overwritten, for the whole run (and shared across every
     * SionnaRtChannelModel instance if you have more than one channel/band
     * -- this is a simulation-wide setting, not per-instance).
     * @param path Output CSV file path.
     */
    static void SetSdtExportPath(const std::string& path);

  protected:

--------------------------------------------------------------------------

Find this (in the protected section, right after CalculateChannelParamsFromPaths's
declaration -- look for GetNumberOfPathsFromCir right after it):

    Ptr<SionnaRtChannelParams> CalculateChannelParamsFromPaths(const py::object paths,
                                                               Ptr<const MobilityModel> aMob,
                                                               Ptr<const MobilityModel> bMob) const;

Change it to (adding one new protected method right after):

    Ptr<SionnaRtChannelParams> CalculateChannelParamsFromPaths(const py::object paths,
                                                               Ptr<const MobilityModel> aMob,
                                                               Ptr<const MobilityModel> bMob) const;

    /**
     * @brief Append one CSV row per path in `paths` to the file configured
     * via SetSdtExportPath(), for an external pipeline to consume live. A
     * no-op if SetSdtExportPath() was never called. See
     * sdt_llm/data/ns3_stream_bridge.py (Python side) for the exact schema
     * this writes and how it's read back -- angles are written in Sionna's
     * own zenith convention with zero conversion on this side, deliberately,
     * so this method stays as mechanical/low-risk as possible.
     * @param paths Python paths object, as returned by CalculatePaths()
     * @param sMob Mobility model of the transmitter endpoint
     * @param uMob Mobility model of the receiver endpoint
     */
    void ExportPathsToSdtCsv(const py::object& paths,
                             Ptr<const MobilityModel> sMob,
                             Ptr<const MobilityModel> uMob) const;

==========================================================================
STEP 2 — src/spectrum/model/sionna-rt-channel-model.cc
==========================================================================

2a) Near the top, check these two includes exist; add whichever are missing
    (harmless to add even if already present via another header):

    #include <fstream>
    #include "ns3/log.h"

2b) Add this anywhere at file scope (e.g. right after the includes, or right
    before the CalculateChannelParamsFromPaths implementation -- exact
    location doesn't matter, it just needs to be outside any function body):

    namespace
    {
    // Output path for SDT CSV export; empty = disabled. File-scope (not a
    // class member) so it's shared across every SionnaRtChannelModel
    // instance -- see SetSdtExportPath's doc comment for why.
    std::string g_sdtExportPath;
    } // anonymous namespace

    void
    SionnaRtChannelModel::SetSdtExportPath(const std::string& path)
    {
        g_sdtExportPath = path;
    }

    void
    SionnaRtChannelModel::ExportPathsToSdtCsv(const py::object& paths,
                                               Ptr<const MobilityModel> sMob,
                                               Ptr<const MobilityModel> uMob) const
    {
        if (g_sdtExportPath.empty())
        {
            return; // export disabled -- SetSdtExportPath() was never called
        }

        static std::ofstream csv;
        if (!csv.is_open())
        {
            csv.open(g_sdtExportPath, std::ios::app);
        }

        py::module_ np = py::module_::import("numpy");

        // Reuse the class's own existing, already-correct extraction
        // helpers for everything except gain -- these are unambiguous,
        // typed, and already used elsewhere in this class.
        MatrixBasedChannelModel::DoubleVector tau = CalculateTauFromPaths(np, paths);
        std::vector<double> doppler = CalculateDopplerFromPaths(np, paths);
        MatrixBasedChannelModel::DoubleVector aoaAz = CalculateAnglesFromPaths(np, paths, "phi_r");
        MatrixBasedChannelModel::DoubleVector aoaZenith = CalculateAnglesFromPaths(np, paths, "theta_r");
        MatrixBasedChannelModel::DoubleVector aodAz = CalculateAnglesFromPaths(np, paths, "phi_t");
        MatrixBasedChannelModel::DoubleVector aodZenith = CalculateAnglesFromPaths(np, paths, "theta_t");

        size_t n = tau.size();
        std::vector<double> gainRe(n, 0.0);
        std::vector<double> gainIm(n, 0.0);

        // Complex path gain: read paths.a directly -- a (real, imag) tuple
        // of numpy arrays. VERIFIED against a live sionna-rt 2.0.1 install
        // (independently of this codebase) to be shape (num_rx, num_rx_ant,
        // num_tx, num_tx_ant, num_paths) -- 5-D -- under the default
        // synthetic_array=True. This is a sionna-rt PACKAGE property, so it
        // should hold regardless of this class's own internal conventions
        // -- but just in case your build sees a different shape (e.g. an
        // extra num_time_steps axis), this is defensive: it logs a warning
        // and writes zero gains rather than crashing, so a shape mismatch
        // is visible in the log instead of silently wrong numbers.
        try
        {
            py::tuple aTuple = paths.attr("a").cast<py::tuple>();
            py::array_t<double> aRe = np.attr("asarray")(aTuple[0]).cast<py::array_t<double>>();
            py::array_t<double> aIm = np.attr("asarray")(aTuple[1]).cast<py::array_t<double>>();
            if (aRe.ndim() == 5)
            {
                auto reBuf = aRe.unchecked<5>();
                auto imBuf = aIm.unchecked<5>();
                size_t availablePaths = static_cast<size_t>(reBuf.shape(4));
                for (size_t i = 0; i < n && i < availablePaths; ++i)
                {
                    gainRe[i] = reBuf(0, 0, 0, 0, i);
                    gainIm[i] = imBuf(0, 0, 0, 0, i);
                }
            }
            else
            {
                NS_LOG_UNCOND("SDT export: paths.a has ndim=" << aRe.ndim()
                              << " (expected 5) -- writing zero gains this update. "
                              << "See ExportPathsToSdtCsv's comment to adjust the indexing.");
            }
        }
        catch (const std::exception& e)
        {
            NS_LOG_UNCOND("SDT export: could not read paths.a (" << e.what()
                          << ") -- writing zero gains this update.");
        }

        Vector txPos = sMob->GetPosition();
        Vector rxPos = uMob->GetPosition();
        double now = Simulator::Now().GetSeconds();
        double fc = GetFrequency();

        for (size_t i = 0; i < n; ++i)
        {
            csv << now << "," << txPos.x << "," << txPos.y << "," << txPos.z << ","
                << rxPos.x << "," << rxPos.y << "," << rxPos.z << "," << fc << ","
                << tau[i] << "," << doppler[i] << ","
                << aoaAz[i] << "," << aoaZenith[i] << "," << aodAz[i] << "," << aodZenith[i] << ","
                << gainRe[i] << "," << gainIm[i] << "\n";
        }
        csv.flush();
    }

2c) Find the implementation of GetNewChannel:

    Ptr<MatrixBasedChannelModel::ChannelMatrix>
    SionnaRtChannelModel::GetNewChannel(py::object paths,
                                         const Ptr<const MobilityModel> sMob,
                                         const Ptr<const MobilityModel> uMob,
                                         Ptr<const PhasedArrayModel> sAntenna,
                                         Ptr<const PhasedArrayModel> uAntenna) const
    {
        <-- add the export call as the VERY FIRST LINE here, before
            anything else in the function -->

Add exactly one line as the first statement in the body:

        ExportPathsToSdtCsv(paths, sMob, uMob);

That's it -- everything after that line is whatever GetNewChannel already
does, untouched.

==========================================================================
STEP 3 — enable it and rebuild
==========================================================================

In your example .cc's main() (e.g. cttc-nr-demo-sionna-rt.cc), near the top,
before Simulator::Run():

    ns3::SionnaRtChannelModel::SetSdtExportPath("/tmp/sdt_radio_stream.csv");

Then:

    cd ~/ns-3-dev
    ./ns3 build
    rm -f /tmp/sdt_radio_stream.csv
    ./ns3 run cttc-nr-demo-sionna-rt

And in another terminal, watch it live:

    cd ~/Downloads/sdt-llm-6g
    source .venv/bin/activate
    python3 scripts/run_ns3_live_demo.py --csv /tmp/sdt_radio_stream.csv --llm-backend mock

==========================================================================
If it doesn't compile
==========================================================================
Paste me the exact compiler error. The most likely spots, in rough order
of likelihood:
  1. `DoubleVector` isn't actually a bare std::vector<double> alias (it's
     declared in matrix-based-channel-model.h, which I haven't seen) --
     if so, the fix is almost always just changing how `tau`/`aoaAz`/etc.
     are indexed (e.g. `tau.at(i)` or a different accessor), not a
     structural change.
  2. `NS_LOG_UNCOND` needs `ns3/log.h` and *some* NS_LOG_COMPONENT_DEFINE
     to exist earlier in the file -- virtually certain to already be there,
     but if not, that's the fix.
  3. `py::array_t<double>` / `.unchecked<5>()` -- needs `pybind11/numpy.h`,
     which sionna-rt-channel-model.h already includes.
