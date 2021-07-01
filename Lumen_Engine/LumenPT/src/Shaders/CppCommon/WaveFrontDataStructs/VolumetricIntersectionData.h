#pragma once
#include "../CudaDefines.h"
#include "../ModelStructs.h"

#include <Optix/optix.h>
#include <Cuda/cuda/helpers.h>
#include <cassert>

#include <nanovdb/NanoVDB.h>

namespace WaveFront
{
	struct VolumetricIntersectionData
	{
		CPU_GPU VolumetricIntersectionData()
			:
			m_RayArrayIndex(0),
			m_EntryT(-1.f),
			m_ExitT(-1.f),
			m_VolumeGrid(nullptr),
			m_PixelIndex({0, 0})
		{
		}

		CPU_GPU VolumetricIntersectionData(
			unsigned int a_RayArrayIndex,
			float a_IntersectionT,
			float a_EntryT,
			float a_ExitT,
			nanovdb::FloatGrid* a_VolumeGrid,
			unsigned int a_PrimitiveIndex,
			unsigned int a_InstanceId,
			const PixelIndex& a_PixelIndex)
			:
			m_RayArrayIndex(a_RayArrayIndex),
			m_EntryT(a_EntryT),
			m_ExitT(a_ExitT),
			m_VolumeGrid(a_VolumeGrid),
			m_PixelIndex(a_PixelIndex)
		{}



		/// <summary> Checks if the data defines an intersection. </summary>
		/// <returns> Returns true if m_IntersectionT is higher than 0.  <em>(boolean)</em> </returns>
		CPU_GPU INLINE bool IsIntersection() const
		{
			return (m_EntryT > 0.f);
		}



		/// <summary>
		/// <b>Description</b> \n The index in the m_Rays array of a RayBatch of the ray the intersection belongs to. \n
		/// <b>Default</b>: 0
		/// </summary>
		unsigned int m_RayArrayIndex;

		//The index of the pixel/surface that this intersection affects.
		PixelIndex m_PixelIndex;

		//Distance along ray to entry of volumetric bounding box
		float m_EntryT;

		//Distance along ray to exit of volumetric bounding box
		float m_ExitT;

		//Pointer to intersected volume as represented by a nanovdb grid of float type
		const nanovdb::FloatGrid* m_VolumeGrid;
	};
}