cmake_minimum_required(VERSION 3.25)

project(exploit)

add_executable(${CMAKE_PROJECT_NAME} main.c utils.c memory.c)

target_link_libraries(${CMAKE_PROJECT_NAME} -static)

# add_custom_command(TARGET ${CMAKE_PROJECT_NAME} COMMAND adb push ${CMAKE_PROJECT_NAME} /data/local/tmp)
# add_custom_command(TARGET ${CMAKE_PROJECT_NAME} COMMAND scp -P 10021 -i /opt/seiji/image/stretch.id_rsa ${CMAKE_PROJECT_NAME} root@localhost:/root)




