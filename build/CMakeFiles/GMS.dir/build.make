# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yfji/Workspace/OpenCV/GMS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yfji/Workspace/OpenCV/GMS/build

# Include any dependencies generated for this target.
include CMakeFiles/GMS.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GMS.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GMS.dir/flags.make

CMakeFiles/GMS.dir/src/main.cpp.o: CMakeFiles/GMS.dir/flags.make
CMakeFiles/GMS.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yfji/Workspace/OpenCV/GMS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GMS.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GMS.dir/src/main.cpp.o -c /home/yfji/Workspace/OpenCV/GMS/src/main.cpp

CMakeFiles/GMS.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GMS.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yfji/Workspace/OpenCV/GMS/src/main.cpp > CMakeFiles/GMS.dir/src/main.cpp.i

CMakeFiles/GMS.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GMS.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yfji/Workspace/OpenCV/GMS/src/main.cpp -o CMakeFiles/GMS.dir/src/main.cpp.s

CMakeFiles/GMS.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/GMS.dir/src/main.cpp.o.requires

CMakeFiles/GMS.dir/src/main.cpp.o.provides: CMakeFiles/GMS.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/GMS.dir/build.make CMakeFiles/GMS.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/GMS.dir/src/main.cpp.o.provides

CMakeFiles/GMS.dir/src/main.cpp.o.provides.build: CMakeFiles/GMS.dir/src/main.cpp.o


CMakeFiles/GMS.dir/src/gms_matcher.cpp.o: CMakeFiles/GMS.dir/flags.make
CMakeFiles/GMS.dir/src/gms_matcher.cpp.o: ../src/gms_matcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yfji/Workspace/OpenCV/GMS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GMS.dir/src/gms_matcher.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GMS.dir/src/gms_matcher.cpp.o -c /home/yfji/Workspace/OpenCV/GMS/src/gms_matcher.cpp

CMakeFiles/GMS.dir/src/gms_matcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GMS.dir/src/gms_matcher.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yfji/Workspace/OpenCV/GMS/src/gms_matcher.cpp > CMakeFiles/GMS.dir/src/gms_matcher.cpp.i

CMakeFiles/GMS.dir/src/gms_matcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GMS.dir/src/gms_matcher.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yfji/Workspace/OpenCV/GMS/src/gms_matcher.cpp -o CMakeFiles/GMS.dir/src/gms_matcher.cpp.s

CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.requires:

.PHONY : CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.requires

CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.provides: CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.requires
	$(MAKE) -f CMakeFiles/GMS.dir/build.make CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.provides.build
.PHONY : CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.provides

CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.provides.build: CMakeFiles/GMS.dir/src/gms_matcher.cpp.o


# Object files for target GMS
GMS_OBJECTS = \
"CMakeFiles/GMS.dir/src/main.cpp.o" \
"CMakeFiles/GMS.dir/src/gms_matcher.cpp.o"

# External object files for target GMS
GMS_EXTERNAL_OBJECTS =

GMS: CMakeFiles/GMS.dir/src/main.cpp.o
GMS: CMakeFiles/GMS.dir/src/gms_matcher.cpp.o
GMS: CMakeFiles/GMS.dir/build.make
GMS: CMakeFiles/GMS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yfji/Workspace/OpenCV/GMS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable GMS"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GMS.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GMS.dir/build: GMS

.PHONY : CMakeFiles/GMS.dir/build

CMakeFiles/GMS.dir/requires: CMakeFiles/GMS.dir/src/main.cpp.o.requires
CMakeFiles/GMS.dir/requires: CMakeFiles/GMS.dir/src/gms_matcher.cpp.o.requires

.PHONY : CMakeFiles/GMS.dir/requires

CMakeFiles/GMS.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GMS.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GMS.dir/clean

CMakeFiles/GMS.dir/depend:
	cd /home/yfji/Workspace/OpenCV/GMS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yfji/Workspace/OpenCV/GMS /home/yfji/Workspace/OpenCV/GMS /home/yfji/Workspace/OpenCV/GMS/build /home/yfji/Workspace/OpenCV/GMS/build /home/yfji/Workspace/OpenCV/GMS/build/CMakeFiles/GMS.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GMS.dir/depend

