#include "Timer.h"

#include <stdexcept>

Timer::Timer()
{
	reset();
}

std::float_t Timer::measure(const TimeUnit format) const
{
	const auto current = std::chrono::high_resolution_clock::now();
	const auto difference = static_cast<std::float_t>(std::chrono::duration_cast<std::chrono::microseconds>(current - begin).count());

	std::float_t formatted = 0;

	switch (format)
	{
	case TimeUnit::SECONDS:
		formatted = difference / 1000000.f;
		break;
	case TimeUnit::MILLIS:
		formatted = difference / 1000.f;
		break;
	case TimeUnit::MICROS:
		formatted = difference;
		break;
	default:
		throw std::runtime_error("Trying to read timer with unimplemented format.");
		break;
	}

	return formatted;
}

void Timer::reset()
{
	begin = std::chrono::high_resolution_clock::now();
}