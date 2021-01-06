#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

// Helped class to allow viewing a vectors data in another format
	// without having to copy the vector

template<typename ViewType, typename VectorType>
class VectorView
{
public:
	VectorView(std::vector<VectorType>& a_Vector);
	VectorView(const std::vector<VectorType>& a_Vector);
	~VectorView();

	uint64_t Size() const;

	ViewType& operator[](uint64_t a_Index);
	const ViewType& operator[](uint64_t a_Index) const;

private:
	std::vector<VectorType> m_Vector;
};

template <typename ViewType, typename VectorType>
VectorView<ViewType, VectorType>::VectorView(std::vector<VectorType>& a_Vector)
	: m_Vector(a_Vector)
{
}

template <typename ViewType, typename VectorType>
VectorView<ViewType, VectorType>::VectorView(const std::vector<VectorType>& a_Vector)
    : m_Vector(a_Vector)
{
}

template <typename ViewType, typename VectorType>
VectorView<ViewType, VectorType>::~VectorView()
{
}

template <typename ViewType, typename VectorType>
uint64_t VectorView<ViewType, VectorType>::Size() const
{
	return (m_Vector.size() * sizeof(VectorType)) / sizeof(ViewType);
}

template <typename ViewType, typename VectorType>
ViewType& VectorView<ViewType, VectorType>::operator[](uint64_t a_Index)
{
	assert(a_Index < Size());
	ViewType* data = reinterpret_cast<ViewType*>(m_Vector.data());

	data += a_Index;

	return *data;
}

template <typename ViewType, typename VectorType>
const ViewType& VectorView<ViewType, VectorType>::operator[](uint64_t a_Index) const
{
	assert(a_Index < Size());
	const ViewType* data = reinterpret_cast<const ViewType*>(m_Vector.data());

	data += a_Index;

	return *data;
}
