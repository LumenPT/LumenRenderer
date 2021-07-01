#include "Profiler.h"

#include "Lumen/Input.h"

#include <queue>

#include "imgui.h"
#include "implot.h"

void Profiler::AddUniqueStats(const FrameStats& a_Stats)
{
    if (m_Recording && a_Stats.m_Id && (m_PreviousFramesStats.empty() || m_PreviousFramesStats.back().m_Id != a_Stats.m_Id))
    {
        m_PreviousFramesStats.push_back(a_Stats);

        if (m_PreviousFramesStats.size() > m_MaxStoredFrames)
        {
            m_PreviousFramesStats.pop_front();
        }

		uint64_t last = 0;
        for (auto& time : a_Stats.m_Times)
        {
            auto& vec = m_ProcessedData[time.first].m_Data;
            if (vec.size() > m_MaxStoredFrames)
            {
				vec.erase(vec.begin());
            }
		
			auto n = last + time.second;
			last = n;
			m_ProcessedData[time.first].m_MaxTime = std::max(m_ProcessedData[time.first].m_MaxTime, n);
			vec.push_back(n);
        }

        if (!m_AutoScroll)
        {
            m_BarOffset += 1.0f;
        }
    }
}

void Profiler::Display()
{
	ImGui::Begin("Profiler");

    if (m_Recording)
    {
        if (ImGui::Button("Pause Recording"))
        {
            m_Recording = false;
        }
    }
    else
    {
        if (ImGui::Button("Continue Recording"))
        {
            m_Recording = true;
        }
    }
    ImGui::SameLine();
    if (m_AutoScroll)
    {
        if (ImGui::Button("Disable Automatic Scrolling"))
        {
            m_AutoScroll = false;
        }
    }
    else
    {
        if (ImGui::Button("Enable Automatic Scrolling"))
        {
            m_AutoScroll = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Chart Pan"))
    {
        m_BarOffset = 0.0f;
        m_AutoScroll = true;
    }

    uint64_t max = 0;
    std::vector<std::string> shown;

    for (auto& processed : m_ProcessedData)
    {
        if (processed.second.m_Display)
        {
            max = std::max(max, processed.second.m_MaxTime);
            shown.push_back(processed.first);
        }
    }

    Plot(shown, max);

    if (ImGui::Button("Enable All"))
        for (auto& d : m_ProcessedData)
            d.second.m_Display = true;

    ImGui::SameLine();
    if (ImGui::Button("Disable All"))
        for (auto& d : m_ProcessedData)
            d.second.m_Display = false;

    if (ImGui::BeginTable("Frame times", 3, ImGuiTableFlags_SizingFixedFit))
    {
        auto id = m_TableStatsId > 0 ? static_cast<uint64_t>(m_TableStatsId) : 0;

        id = std::min(id, m_PreviousFramesStats.size() - 1);
        ImGui::TableNextColumn();
        ImGui::Text(" ");
        ImGui::TableNextColumn();
        ImGui::Text("Name");
        ImGui::TableNextColumn();
        ImGui::Text("Time (microseconds)");

        uint64_t totalSelected = 0;
        uint64_t total = 0;

        for (auto& stats : m_ProcessedData)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(("Checkbox" + stats.first).c_str());
            ImGui::Checkbox("", &stats.second.m_Display);
            ImGui::PopID();
            ImGui::TableNextColumn();
            ImGui::Text(stats.first.c_str());
            ImGui::TableNextColumn();

            auto n = m_PreviousFramesStats[id].m_Times[stats.first];
            ImGui::Text("%llu", n);

            total += n;
            if (stats.second.m_Display)
                totalSelected += n;

        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::Text("Total (Displayed)");
        ImGui::Text("Total");
        ImGui::TableNextColumn();
        ImGui::Text("%llu", totalSelected);
        ImGui::Text("%llu", total);

        ImGui::EndTable();
    }

	ImGui::End();
}
void Profiler::Plot(std::vector<std::string>& a_Shown, uint64_t a_MaxY)
{
    if (a_Shown.size() != m_ProcessedData.size())
    {
        auto startId = GetStartingId();
        startId = std::max(startId, 0ll);

        std::map<std::string, ProcessedData> processed;
        std::unordered_map<uint64_t, uint64_t> prev;
        uint64_t max = 0;
        for (auto& name : a_Shown)
        {
            auto& data = processed[name];
            data.m_Data = std::vector<uint64_t>(startId);
            data.m_Data.reserve(data.m_Data.size() + m_BarsDisplayed);
            for (auto i = startId; i < m_PreviousFramesStats.size(); i++)
            {
                auto& stats = m_PreviousFramesStats[i];

                prev[i] = prev[i] + stats.m_Times[name];
                data.m_Data.push_back(prev[i]);
                max = std::max(max, prev[i]);
            }            
        }
        ImPlot::SetNextPlotLimitsY(0.0, static_cast<double>(max) * 1.1, ImGuiCond_Always);
        ImPlot::SetNextPlotLimitsX(static_cast<double>(GetStartingId()), static_cast<double>(m_PreviousFramesStats.size()) - m_BarOffset, ImGuiCond_Always);

        if (ImPlot::BeginPlot("Frame Times", "Frame", "Time (microseconds)", ImVec2(-1, 0), ImPlotFlags_Crosshairs))
        {
            for (auto iter = processed.crbegin(); iter != processed.crend(); ++iter)
            {
                auto pair = *iter;
                if (pair.second.m_Display)
                    ImPlot::PlotBars(pair.first.c_str(), pair.second.m_Data.data(), pair.second.m_Data.size(), 1.0f);
            }
            if (ImGui::IsMouseClicked(0))
                m_TableStatsId = (ImPlot::GetPlotMousePos().x);

            if (ImPlot::IsPlotHovered())
            {
                auto wheel = Lumen::Input::GetMouseWheel().y;
                m_BarsDisplayed += -wheel * 3.0f;
                if (Lumen::Input::IsMouseButtonPressed(0)) // LMB
                {
                    m_BarOffset += m_BarsDisplayed / 1000.0f * -Lumen::Input::GetMouseDeltaX();
                    m_BarOffset = std::max(m_BarOffset, 0.0f);
                }
            }

            ImPlot::EndPlot();
        }

    }
    else
    {
        ImPlot::SetNextPlotLimitsY(0.0, static_cast<double>(a_MaxY) * 1.1, ImGuiCond_Always);
        ImPlot::SetNextPlotLimitsX(static_cast<double>(m_PreviousFramesStats.size()) - m_BarsDisplayed - (m_BarOffset),
            static_cast<double>(m_PreviousFramesStats.size()) - m_BarOffset, ImGuiCond_Always);
    
        if (ImPlot::BeginPlot("Frame Times", "Frame", "Time (microseconds)", ImVec2(-1, 0), ImPlotFlags_Crosshairs))
        {
            for (auto iter = m_ProcessedData.crbegin(); iter != m_ProcessedData.crend(); ++iter)
            {
                auto pair = *iter;
                if (pair.second.m_Display)
                    ImPlot::PlotBars(pair.first.c_str(), pair.second.m_Data.data(), pair.second.m_Data.size(), 1.0f);
            }
            if (ImGui::IsMouseClicked(0))
                m_TableStatsId = (ImPlot::GetPlotMousePos().x);

            if (ImPlot::IsPlotHovered())
            {
                auto wheel = Lumen::Input::GetMouseWheel().y;
                m_BarsDisplayed += -wheel * 3.0f;
                printf("Profiler %f", wheel);
                if (Lumen::Input::IsMouseButtonPressed(0)) // LMB
                {
                    m_BarOffset += m_BarsDisplayed / 1000.0f * -Lumen::Input::GetMouseDeltaX();
                    m_BarOffset = std::max(m_BarOffset, 0.0f);
                }
            }
            

            ImPlot::EndPlot();
        }        
    }
    m_TableStatsId = m_TableStatsId > 0 ? static_cast<uint64_t>(m_TableStatsId) : 0;
    m_TableStatsId = std::min(m_TableStatsId, m_PreviousFramesStats.size() - 1);
}

int64_t Profiler::GetStartingId() const
{
    return m_PreviousFramesStats.size() - m_BarsDisplayed - static_cast<uint64_t>(m_BarOffset);
}
