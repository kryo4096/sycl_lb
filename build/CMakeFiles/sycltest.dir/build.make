# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jonas/Code/C++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jonas/Code/C++/build

# Include any dependencies generated for this target.
include CMakeFiles/sycltest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sycltest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sycltest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sycltest.dir/flags.make

CMakeFiles/sycltest.dir/main.cpp.o: CMakeFiles/sycltest.dir/flags.make
CMakeFiles/sycltest.dir/main.cpp.o: ../main.cpp
CMakeFiles/sycltest.dir/main.cpp.o: CMakeFiles/sycltest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jonas/Code/C++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sycltest.dir/main.cpp.o"
	/usr/local/bin/syclcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sycltest.dir/main.cpp.o -MF CMakeFiles/sycltest.dir/main.cpp.o.d -o CMakeFiles/sycltest.dir/main.cpp.o -c /home/jonas/Code/C++/main.cpp

CMakeFiles/sycltest.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sycltest.dir/main.cpp.i"
	/usr/local/bin/syclcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jonas/Code/C++/main.cpp > CMakeFiles/sycltest.dir/main.cpp.i

CMakeFiles/sycltest.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sycltest.dir/main.cpp.s"
	/usr/local/bin/syclcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jonas/Code/C++/main.cpp -o CMakeFiles/sycltest.dir/main.cpp.s

# Object files for target sycltest
sycltest_OBJECTS = \
"CMakeFiles/sycltest.dir/main.cpp.o"

# External object files for target sycltest
sycltest_EXTERNAL_OBJECTS =

sycltest: CMakeFiles/sycltest.dir/main.cpp.o
sycltest: CMakeFiles/sycltest.dir/build.make
sycltest: /usr/lib/libpng.so
sycltest: /usr/lib/libz.so
sycltest: CMakeFiles/sycltest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jonas/Code/C++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sycltest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sycltest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sycltest.dir/build: sycltest
.PHONY : CMakeFiles/sycltest.dir/build

CMakeFiles/sycltest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sycltest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sycltest.dir/clean

CMakeFiles/sycltest.dir/depend:
	cd /home/jonas/Code/C++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jonas/Code/C++ /home/jonas/Code/C++ /home/jonas/Code/C++/build /home/jonas/Code/C++/build /home/jonas/Code/C++/build/CMakeFiles/sycltest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sycltest.dir/depend

