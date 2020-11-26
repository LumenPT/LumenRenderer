#pragma once


#ifdef LMN_PLATFORM_WINDOWS

extern Lumen::LumenApp* Lumen::CreateApplication();

int main(int argc, char** argv)
{
	Lumen::Log::Init();
#include "lmnpch.h"
#include "lmnpch.h"

	LMN_CORE_WARN("Initialized logger!");
	int a = 5;
	LMN_INFO("Hello log! Var={0}", a);
	
	auto app = Lumen::CreateApplication();
	app->Run();
	delete app;
}

#endif
