#pragma once

#include "Lumen/Renderer/LumenRenderer.h"


#include <deque>
class Profiler
{
public:
    Profiler()
        : m_BarsDisplayed(300)
        , m_BarOffset(0.0f) 
        , m_Recording(true)
        , m_AutoScroll(true) {}

    void AddUniqueStats(const FrameStats& a_Stats);

    void Display();

private:
    void Plot(std::vector<std::string>& a_Shown, uint64_t a_MaxY);

    int64_t GetStartingId() const;

    struct ProcessedData
    {
        std::vector<uint64_t> m_Data;
        bool m_Display = true;
        uint64_t m_MaxTime;
    };

    std::deque<FrameStats> m_PreviousFramesStats;
    std::map<std::string, ProcessedData> m_ProcessedData;
    const uint32_t m_MaxStoredFrames = 5 * 60 * 60; // 5 minutes of running at 60FPS
    uint32_t m_BarsDisplayed;
    float m_BarOffset;

    bool m_Recording;
    bool m_AutoScroll;
    uint64_t m_TableStatsId;
};

