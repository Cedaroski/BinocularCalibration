# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /home/user/Install/clion-2018.3.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/user/Install/clion-2018.3.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user/Binocular/BinocularCalibration

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/Binocular/BinocularCalibration/cmake-build-debug

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/user/Install/clion-2018.3.2/bin/cmake/linux/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/user/Install/clion-2018.3.2/bin/cmake/linux/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/user/Binocular/BinocularCalibration/cmake-build-debug/CMakeFiles /home/user/Binocular/BinocularCalibration/cmake-build-debug/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/user/Binocular/BinocularCalibration/cmake-build-debug/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named BinocularCalibration

# Build rule for target.
BinocularCalibration: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 BinocularCalibration
.PHONY : BinocularCalibration

# fast build rule for target.
BinocularCalibration/fast:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/build
.PHONY : BinocularCalibration/fast

# target to build an object file
StereoCalib.o:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/StereoCalib.o
.PHONY : StereoCalib.o

# target to preprocess a source file
StereoCalib.i:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/StereoCalib.i
.PHONY : StereoCalib.i

# target to generate assembly for a file
StereoCalib.s:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/StereoCalib.s
.PHONY : StereoCalib.s

# target to build an object file
opencv_contrib.o:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/opencv_contrib.o
.PHONY : opencv_contrib.o

# target to preprocess a source file
opencv_contrib.i:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/opencv_contrib.i
.PHONY : opencv_contrib.i

# target to generate assembly for a file
opencv_contrib.s:
	$(MAKE) -f CMakeFiles/BinocularCalibration.dir/build.make CMakeFiles/BinocularCalibration.dir/opencv_contrib.s
.PHONY : opencv_contrib.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... BinocularCalibration"
	@echo "... edit_cache"
	@echo "... StereoCalib.o"
	@echo "... StereoCalib.i"
	@echo "... StereoCalib.s"
	@echo "... opencv_contrib.o"
	@echo "... opencv_contrib.i"
	@echo "... opencv_contrib.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

