
# top level CMakeLists.txt
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8)

#SET(TARGET "step-0-material")
#SET(TARGET "step-1-elastic")
#SET(TARGET "step-2-piezoelectric")
SET(TARGET "aphex")

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

DEAL_II_INITIALIZE_CACHED_VARIABLES ()

PROJECT (mandy)
SET (MANDY_VERSION_MAJOR 0)
SET (MANDY_VERSION_MINOR 1)
SET (MANDY_VERSION_SUBMINOR 0)
SET (MANDY_VERSION_PATCH 0)

ADD_SUBDIRECTORY(source)

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
  )

DEAL_II_SETUP_TARGET (mandy)
INCLUDE_DIRECTORIES(include
  ${CMAKE_SOURCE_DIR}/mandy/include
  )

ADD_EXECUTABLE (${TARGET} ${TARGET}.cc)
DEAL_II_SETUP_TARGET (${TARGET})
TARGET_LINK_LIBRARIES (${TARGET} mandy)



