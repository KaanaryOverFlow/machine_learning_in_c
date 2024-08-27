if [ -n "$1" ];
then
    C=`locate toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$1-clang | head -1`
    CXX=`locate toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$1-clang | tail -1`
    echo "Configuring for arm64"
    echo "C compiler : $C"
    echo "C++ compiler : $CXX"
else
	C=clang
	CXX=clang++
fi

mkdir -p ../build
echo "all:" > ../build/Makefile
echo -e "\t$C ../src/*.c -static -o exploit" >> ../build/Makefile
echo -e "\t# adb push exploit /data/local/tmp" >> ../build/Makefile
echo -e "\t# scp ./exploit root@localhost:/root/" >> ../build/Makefile
