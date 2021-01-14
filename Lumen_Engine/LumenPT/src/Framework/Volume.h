#pragma once

#include <string>

#include <nanovdb/NanoVDB.h>

class Volume
{
public:
	Volume();
	~Volume();

	void Load(const std::string& a_FilePath);
	
private:
	
};