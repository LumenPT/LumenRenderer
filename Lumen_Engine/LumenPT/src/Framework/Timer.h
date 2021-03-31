#pragma once
#include <chrono>
#include <cmath>

/*
 * Used to indicate the desired time format returned by measure.
 */
enum class TimeUnit
{
	SECONDS,
	MILLIS,
	MICROS
};

/*
 * Timer is a simple utility to measure time passing.
 * When a timer is constructed, it measures the current time.
 * The time since that moment can then be measured.
 * 
 * Time can also be reset.
 */
class Timer
{
public:
	Timer();

public:
	/*
	 * Measure the time that has passed since this timer was reset.
	 * The provided format determines the time unit used.
	 */
	std::float_t measure(const TimeUnit format) const;

	/*
	 * Reset the timer to measure from the time this function was called.
	 */
	void reset();

private:
	std::chrono::high_resolution_clock::time_point begin;
};