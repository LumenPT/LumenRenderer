#include "Volume.h"

#include <cassert>

#include <nanovdb/util/IO.h>

Volume::Volume()
{
	
}

Volume::~Volume()
{
	
}

void Volume::Load(const std::string& a_FilePath)
{
    auto handle = nanovdb::io::readGrid(a_FilePath); // reads first grid from file

    auto* grid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float

    if (!grid)
    {
        assert(false); //File did not contain a grid with value type float
    }

    auto acc = grid->getAccessor(); // create an accessor for fast access to multiple values
    for (int i = 97; i < 104; ++i) 
    {
        printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, acc.getValue(nanovdb::Coord(i, 0, 0)));
    }
    
}
