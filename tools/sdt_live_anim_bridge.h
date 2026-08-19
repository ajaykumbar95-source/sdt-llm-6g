#pragma once

#include "ns3/animation-interface.h"
#include "ns3/nr-phy-mac-common.h"
#include "ns3/simulator.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace ns3
{

class SdtLiveAnimBridge
{
  public:
    static void Configure(AnimationInterface& anim, const std::string& outputFile)
    {
        const char* enabled = std::getenv("SDT_LIVE_DEMO");
        if (!enabled || std::string(enabled) != "1")
        {
            return;
        }

        s_enabled = true;
        s_outputFile = outputFile;

        const char* speed = std::getenv("SDT_LIVE_SPEED");
        if (speed)
        {
            try
            {
                s_speed = std::stod(speed);
            }
            catch (...)
            {
                s_speed = 1.0;
            }
        }

        if (!(s_speed > 0.0))
        {
            s_speed = 1.0;
        }

        s_stream.open(s_outputFile, std::ios::out | std::ios::trunc);
        if (!s_stream.is_open())
        {
            NS_LOG_WARN("Could not open live visualization stream: " << s_outputFile);
            s_enabled = false;
            return;
        }

        s_wallStart = std::chrono::steady_clock::now();
        s_simStart = Simulator::Now().GetSeconds();

        s_stream << "{\"type\":\"live_start\",\"sim_time\":"
                 << std::fixed << std::setprecision(9)
                 << Simulator::Now().GetSeconds()
                 << ",\"speed\":" << s_speed << "}\n"
                 << std::flush;

        anim.SetAnimWriteCallback(&SdtLiveAnimBridge::OnAnimationWrite);

        Simulator::Schedule(MilliSeconds(10), &SdtLiveAnimBridge::Heartbeat);

        NS_LOG_UNCOND("SDT live visualizer enabled: " << s_outputFile
                                                       << " speed=" << s_speed << "x");
    }

    static void Close()
    {
        if (!s_enabled)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(s_mutex);

        s_stream << "{\"type\":\"live_end\",\"sim_time\":"
                 << std::fixed << std::setprecision(9)
                 << Simulator::Now().GetSeconds()
                 << "}\n"
                 << std::flush;

        s_stream.close();
        s_enabled = false;
    }

  public:
    /**
     * Real 5G-LENA UE-side NR reception event.
     *
     * The current scenario has one gNB, node 0. The UE node ID is extracted
     * from the Config path (/NodeList/<id>/...).
     */
    static void OnNrUeRx(std::string path, RxPacketTraceParams params)
    {
        if (!s_enabled)
        {
            return;
        }

        uint32_t ueNodeId = ExtractNodeId(path);

        PaceToSimulationTime();

        std::lock_guard<std::mutex> lock(s_mutex);

        s_stream
            << "{\"type\":\"nr_packet\","
            << "\"direction\":\"DL\","
            << "\"from\":0,"
            << "\"to\":" << ueNodeId << ","
            << "\"rnti\":" << params.m_rnti << ","
            << "\"cell_id\":" << params.m_cellId << ","
            << "\"bwp_id\":" << params.m_bwpId << ","
            << "\"tb_size\":" << params.m_tbSize << ","
            << "\"sinr\":" << std::setprecision(9) << params.m_sinr << ","
            << "\"corrupt\":" << (params.m_corrupt ? "true" : "false") << ","
            << "\"sim_time\":" << Simulator::Now().GetSeconds()
            << "}\n"
            << std::flush;
    }

  private:
    static uint32_t ExtractNodeId(const std::string& path)
    {
        const std::string marker = "/NodeList/";
        auto pos = path.find(marker);

        if (pos == std::string::npos)
        {
            return 0;
        }

        pos += marker.size();

        auto end = path.find('/', pos);
        if (end == std::string::npos)
        {
            return 0;
        }

        try
        {
            return static_cast<uint32_t>(
                std::stoul(path.substr(pos, end - pos))
            );
        }
        catch (...)
        {
            return 0;
        }
    }

    static std::string JsonEscape(const std::string& input)
    {
        std::ostringstream out;
        for (char c : input)
        {
            switch (c)
            {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << c;
                break;
            }
        }
        return out.str();
    }

    static void PaceToSimulationTime()
    {
        if (!s_enabled)
        {
            return;
        }

        const double simNow = Simulator::Now().GetSeconds();
        const double elapsedSim = simNow - s_simStart;
        const double targetWallSeconds = elapsedSim / s_speed;

        const auto target =
            s_wallStart +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(targetWallSeconds));

        const auto now = std::chrono::steady_clock::now();

        if (target > now)
        {
            std::this_thread::sleep_for(target - now);
        }
    }

    static void Heartbeat()
    {
        if (!s_enabled || Simulator::IsFinished())
        {
            return;
        }

        PaceToSimulationTime();

        {
            std::lock_guard<std::mutex> lock(s_mutex);
            s_stream << "{\"type\":\"heartbeat\",\"sim_time\":"
                     << std::fixed << std::setprecision(9)
                     << Simulator::Now().GetSeconds()
                     << "}\n"
                     << std::flush;
        }

        Simulator::Schedule(MilliSeconds(10), &SdtLiveAnimBridge::Heartbeat);
    }

    static void OnAnimationWrite(const char* text)
    {
        if (!s_enabled || !text)
        {
            return;
        }

        PaceToSimulationTime();

        std::lock_guard<std::mutex> lock(s_mutex);

        s_stream << "{\"type\":\"anim\",\"sim_time\":"
                 << std::fixed << std::setprecision(9)
                 << Simulator::Now().GetSeconds()
                 << ",\"xml\":\""
                 << JsonEscape(text)
                 << "\"}\n"
                 << std::flush;
    }

    inline static bool s_enabled = false;
    inline static double s_speed = 1.0;
    inline static double s_simStart = 0.0;
    inline static std::chrono::steady_clock::time_point s_wallStart;
    inline static std::string s_outputFile;
    inline static std::ofstream s_stream;
    inline static std::mutex s_mutex;
};

} // namespace ns3
