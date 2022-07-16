echo "Download MS-COCO 2014 images data ..."
cd text_to_images_models/data/coco && wget http://images.cocodataset.org/zips/train2014.zip && cd ../../../
cd text_to_images_models/data/coco && wget http://images.cocodataset.org/zips/val2014.zip && cd ../../../
echo "Extracting MS-COCO 2014 images data ..."
cd text_to_images_models/data/coco && unzip train2014.zip && rm -rf train2014.zip && cd ../../../
cd text_to_images_models/data/coco && unzip val2014.zip && rm -rf val2014.zip && cd ../../../
echo "Done."