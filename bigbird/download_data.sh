if [ $(basename "$PWD") != "intrinsic-neural-fields" ]
then
	echo "Please call this script from the intrinsic-neural-fields directory."
	exit 1
fi

if [ ! -d ".data" ]
then
	echo "The directory data does not exist. Either (a) create it or (b) call download_data.sh first."
	exit 1
fi

echo "Downloading and unzipping data to data/bigbird"

wget "https://vision.in.tum.de/webshare/g/intrinsic-neural-fields/data/bigbird.zip" -P "data"
unzip "data/bigbird.zip" -d "data"
rm -rf "data/bigbird.zip"