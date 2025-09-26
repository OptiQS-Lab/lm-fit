# Levenberg Marquardt curve fitting on CPU/CUDA 
# https://github.com/OptiQS-Lab/lm-fit
# see also CMake configuration in /docs/installation.rst

# CMake
 
cmake_minimum_required( VERSION 3.22.1 )
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

if( NOT PROJECT_NAME )
  project( lm-fit VERSION 0.0.1 )  # TODO: Use git tag versioning
  include( CTest )
endif()

if( NOT CMAKE_CXX_STANDARD )
	set( CMAKE_CXX_STANDARD 14 )
endif()

if( MSVC ) # link runtime statically with MSVC
  foreach( type ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} )
    string( TOUPPER ${type} TYPE )
    foreach( flags CMAKE_C_FLAGS_${TYPE} CMAKE_CXX_FLAGS_${TYPE} )
      get_property( help CACHE ${flags} PROPERTY HELPSTRING )
      string( REPLACE "/MD" "/MT" ${flags} "${${flags}}" )
      set( ${flags} "${${flags}}" CACHE STRING "${help}" FORCE )
    endforeach()
  endforeach()
endif()

function( add_launcher target executable arguments working_directory )
  if( MSVC12 OR MSVC14 )
    file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/${target}.vcxproj.user
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
"<Project ToolsVersion=\"14.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n"
"  <PropertyGroup>\n"
"    <LocalDebuggerCommand>${executable}</LocalDebuggerCommand>\n"
"    <LocalDebuggerCommandArguments>${arguments}</LocalDebuggerCommandArguments>\n"
"    <LocalDebuggerWorkingDirectory>${working_directory}</LocalDebuggerWorkingDirectory>\n"
"  </PropertyGroup>\n"
"</Project>\n"
    )
  endif()
endfunction()

# Default values

# Set the default value for the CUDA architecture flag 
set( DEFAULT_CUDA_ARCH All )

# Set the default value for the USE_CUBLAS flag.  Valid values are "ON" or "OFF".
set( DEFAULT_USE_CUBLAS OFF )

# Set the default value for the precision
set( DEFAULT_DOUBLE_PRECISION OFF )

# Precision

#set( DOUBLE_PRECISION ${DEFAULT_DOUBLE_PRECISION} )

if( DEFINED DOUBLE_PRECISION AND DOUBLE_PRECISION)
    add_definitions( -DGPUFIT_DOUBLE )
endif()

# Boost

find_package( Boost 1.58.0 )
if( Boost_FOUND )
  function( add_boost_test modules name )
    string( REPLACE ";" "_" prefix "${modules}" )
    set( target ${prefix}_Test_${name} )
    add_executable( ${target} ${name}.cpp
      ${PROJECT_SOURCE_DIR}/tests/utils.h
      ${PROJECT_SOURCE_DIR}/tests/utils.cpp
    )
    target_include_directories( ${target} PRIVATE ${PROJECT_SOURCE_DIR} )
    target_link_libraries( ${target} ${modules} Boost::boost )
    set_property( TARGET ${target}
      PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" )
    set_property( TARGET ${target} PROPERTY FOLDER Tests )

    add_test( NAME ${target}
      COMMAND ${target} --build_info --log_level=all --report_level=detailed )
  endfunction()
else()
  set( BUILD_TESTING OFF )
  message( WARNING "Boost NOT found - skipping tests! (set BOOST_ROOT manually)" )
endif()

# CPU

add_subdirectory( lm-fit/cpu )

# CUDA

add_subdirectory( lm-fit/cuda )

# # Examples using CPU or CUDA

add_subdirectory( examples/c++/ )

# # Tests

# if( BUILD_TESTING )
#   add_subdirectory( tests )
# endif()

