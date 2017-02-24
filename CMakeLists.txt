
# top level CMakeLists.txt
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8)

SET (TARGET "aphex")

FIND_PACKAGE (deal.II 8.5.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF (NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF ()

MESSAGE(STATUS "Found deal.II version ${DEAL_II_PACKAGE_VERSION} at '${deal.II_DIR}'")

# Configure libmandy
SET (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

IF (EXISTS ${CMAKE_SOURCE_DIR}/doc/CMakeLists.txt)
  SET (MANDY_HAVE_DOC_DIRECTORY TRUE)
ENDIF ()

DEAL_II_INITIALIZE_CACHED_VARIABLES ()

PROJECT (mandy-${MANDY_VERSION_MAJOR} CXX C)

SET (MANDY_VERSION_MAJOR 0)
SET (MANDY_VERSION_MINOR 1)
SET (MANDY_VERSION_SUBMINOR 0)
SET (MANDY_VERSION_PATCH 0)

SET (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

ADD_SUBDIRECTORY (source)

INCLUDE_DIRECTORIES (include/mandy)

ADD_LIBRARY(mandy
  # tools
  source/matrix_creator.cc
  source/vector_creator.cc
  # physics
  source/dielectric_tensor.cc
  source/elastic_tensor.cc
  source/lattice_tensor.cc
  source/piezoelectric_tensor.cc
  source/polarelectric_tensor.cc
  source/tensor_base.cc
  # tools
  source/function_tools.cc
  # base models
  source/elastic_problem.cc
  source/piezoelectric_problem.cc
  )

DEAL_II_SETUP_TARGET (mandy)
INCLUDE_DIRECTORIES (include
  ${CMAKE_SOURCE_DIR}/mandy/include
  )

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET (debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET (release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

# Define a distclean target to remove every generated file:
ADD_CUSTOM_TARGET(distclean
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
  COMMAND ${CMAKE_COMMAND} -E remove CMakeCache.txt cmake_install.cmake Makefile
  COMMENT "Run distclean"
  )

ADD_EXECUTABLE (${TARGET} ${TARGET}.cc)
DEAL_II_SETUP_TARGET (${TARGET})
TARGET_LINK_LIBRARIES (${TARGET} mandy)



