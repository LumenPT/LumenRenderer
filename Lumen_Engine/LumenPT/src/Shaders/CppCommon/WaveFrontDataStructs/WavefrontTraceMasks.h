#pragma once
#include <Optix/optix_types.h>

namespace WaveFront
{

    enum TraceMaskType
    {
        SOLIDS = OptixVisibilityMask(0b00000001),
        VOLUMES = OptixVisibilityMask(0b00000010),
        ALL = SOLIDS | VOLUMES
    };

}
