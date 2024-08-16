cmake_minimum_required(VERSION 3.12)
project(PDEProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

find_package(deal.II 9.3.1 REQUIRED
  COMPONENTS Trilinos
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
    src/setup.cpp
    src/functions.cpp
    src/operator.cpp
    main.cpp
)

add_executable(PDEProject ${SOURCES})

deal_ii_setup_target(PDEProject)
