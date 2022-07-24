# CMake generated Testfile for 
# Source directory: D:/bld/rdkit_1657061356439/work/rdkit
# Build directory: D:/bld/rdkit_1657061356439/work/rdkit
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pythonTestDirRoot "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit")
set_tests_properties(pythonTestDirRoot PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;16;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
add_test(pythonTestDirML "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/ML/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit/ML")
set_tests_properties(pythonTestDirML PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;18;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
add_test(pythonTestDirDataStructs "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/DataStructs/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit/DataStructs")
set_tests_properties(pythonTestDirDataStructs PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;20;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
add_test(pythonTestDirDbase "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/Dbase/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit/Dbase")
set_tests_properties(pythonTestDirDbase PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;22;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
add_test(pythonTestDirSimDivFilters "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/SimDivFilters/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit/SimDivFilters")
set_tests_properties(pythonTestDirSimDivFilters PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;24;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
add_test(pythonTestDirVLib "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/VLib/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit/VLib")
set_tests_properties(pythonTestDirVLib PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;26;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
add_test(pythonTestSping "C:/Users/theco/anaconda3/envs/deeplearning/python.exe" "D:/bld/rdkit_1657061356439/work/rdkit/Chem/test_list.py" "--testDir" "D:/bld/rdkit_1657061356439/work/rdkit/sping")
set_tests_properties(pythonTestSping PROPERTIES  _BACKTRACE_TRIPLES "D:/bld/rdkit_1657061356439/work/Code/cmake/Modules/RDKitUtils.cmake;212;add_test;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;28;add_pytest;D:/bld/rdkit_1657061356439/work/rdkit/CMakeLists.txt;0;")
subdirs("Chem")
