# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yeongjunkim/projects/tfliteDriver

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/tfliteDriver.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tfliteDriver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tfliteDriver.dir/flags.make

CMakeFiles/tfliteDriver.dir/main.cpp.o: CMakeFiles/tfliteDriver.dir/flags.make
CMakeFiles/tfliteDriver.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tfliteDriver.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tfliteDriver.dir/main.cpp.o -c /Users/yeongjunkim/projects/tfliteDriver/main.cpp

CMakeFiles/tfliteDriver.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tfliteDriver.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yeongjunkim/projects/tfliteDriver/main.cpp > CMakeFiles/tfliteDriver.dir/main.cpp.i

CMakeFiles/tfliteDriver.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tfliteDriver.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yeongjunkim/projects/tfliteDriver/main.cpp -o CMakeFiles/tfliteDriver.dir/main.cpp.s

CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.o: CMakeFiles/tfliteDriver.dir/flags.make
CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.o: ../NNModelParser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.o -c /Users/yeongjunkim/projects/tfliteDriver/NNModelParser.cpp

CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yeongjunkim/projects/tfliteDriver/NNModelParser.cpp > CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.i

CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yeongjunkim/projects/tfliteDriver/NNModelParser.cpp -o CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.s

# Object files for target tfliteDriver
tfliteDriver_OBJECTS = \
"CMakeFiles/tfliteDriver.dir/main.cpp.o" \
"CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.o"

# External object files for target tfliteDriver
tfliteDriver_EXTERNAL_OBJECTS =

tfliteDriver: CMakeFiles/tfliteDriver.dir/main.cpp.o
tfliteDriver: CMakeFiles/tfliteDriver.dir/NNModelParser.cpp.o
tfliteDriver: CMakeFiles/tfliteDriver.dir/build.make
tfliteDriver: CMakeFiles/tfliteDriver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable tfliteDriver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tfliteDriver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tfliteDriver.dir/build: tfliteDriver

.PHONY : CMakeFiles/tfliteDriver.dir/build

CMakeFiles/tfliteDriver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tfliteDriver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tfliteDriver.dir/clean

CMakeFiles/tfliteDriver.dir/depend:
	cd /Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yeongjunkim/projects/tfliteDriver /Users/yeongjunkim/projects/tfliteDriver /Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug /Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug /Users/yeongjunkim/projects/tfliteDriver/cmake-build-debug/CMakeFiles/tfliteDriver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tfliteDriver.dir/depend
