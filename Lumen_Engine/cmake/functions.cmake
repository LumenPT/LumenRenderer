function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

#function to transform input paths to output paths.
function(changePath REGEX_MATCH REGEX_REPLACE RESULT_VAR_NAME)

	set(RESULT "")

	foreach(ARG IN LISTS ARGN)

		string(REGEX REPLACE ${REGEX_MATCH} ${REGEX_REPLACE} OUTPUTPATH ${ARG})
		list(APPEND RESULT ${OUTPUTPATH})

		#message("\n Replaced: ${ARG} \n with: ${OUTPUTPATH}")

	endforeach()

	set(${RESULT_VAR_NAME} ${RESULT} PARENT_SCOPE)
	
endfunction()

function(printList)

    foreach(ARG IN LISTS ARGN)

        message("${ARG}")

    endforeach()

endfunction()