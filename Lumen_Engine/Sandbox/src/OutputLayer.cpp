#include "OutputLayer.h"

#include "LumenPT.h"

#include "Glad/glad.h"

#include <iostream>

OutputLayer::OutputLayer()
{
	auto vs = glCreateShader(GL_VERTEX_SHADER);
	auto fs = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vs, 1, &m_VSSource, nullptr);
	glCompileShader(vs);

	int success;
	char infoLog[512];
	glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vs, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	glShaderSource(fs, 1, &m_FSSource, nullptr);
	glCompileShader(fs);
	glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fs, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	auto program = glCreateProgram();

	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cout << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

	glDeleteShader(vs);
	glDeleteShader(fs);

	m_Program = program;

	LumenPT::InitializationData init;

	m_LumenPT = std::make_unique<LumenPT>(init);



}

OutputLayer::~OutputLayer()
{
	glDeleteProgram(m_Program);
}

void OutputLayer::OnUpdate(){

	auto texture = m_LumenPT->TraceFrame(); // TRACE SUM

	glBindTexture(GL_TEXTURE_2D, texture);
	glUseProgram(m_Program);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}
