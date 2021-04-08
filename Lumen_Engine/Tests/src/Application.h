#pragma once
#include "OutputLayer.h"
#include <Lumen.h>

class Tests : public Lumen::LumenApp
{

public:

    Tests();
    ~Tests();

private:

    bool Init();

    std::unique_ptr<OutputLayer> m_OutputLayer;

};