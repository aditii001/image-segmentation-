# image-segmentation-
image processing
disp('1. Thresholding Method by Otsu''s Method');
disp('2. Color-Based Segmentation such as K-Means clustering');
disp('3. Transform methods as watershed segmentation');
disp('4. Texture method such as texture filters');
disp(' Choose which segmentation to perform :');
n=input('Enter the number: ');
switch n
case 1
I=imread('rice.png');
subplot(2,4,1), imshow(I), title('Original Image');
se=strel('disk',15);
background=imopen(I,se); % morphological opening
subplot(2,4,2), imshow(background), title('Morphological Opening');
I2=I-background; %subtracts original background of the image with the morphological opened background
subplot(2,4,3), imshow(I2), title('B - O Image');
I3=imadjust(I2); % increase the contrast of the image ( 1% in both low and high intensities)
subplot(2,4,4), imshow(I3), title('Contrast Increase');
bw=imbinarize(I3); % converts greyscale to binary image
bw=bwareaopen(bw,50); % removes background noise from the image
subplot(2,4,5), imshow(bw), title('Binary Version');
cc=bwconncomp(bw,4) % connected components
p=cc.NumObjects % total no of objects
prompt='Enter label of grain to view ?';
x=input(prompt)
if x<0 and x>p
disp('Not a valid Number');
end
grain=false(size(bw));
grain(cc.PixelIdxList{x})=true;
subplot(2,4,6), imshow(grain), title('Rice Grain Label');
labeled=labelmatrix(cc);
whos labeled
RGB_label=label2rgb(labeled,'spring','c','shuffle');
subplot(2,4,7), imshow(RGB_label), title('Color-Mapped');
case 2
he = imread('hestain.png');
subplot(2,4,1),imshow(he), title('H&E image');
lab_he = rgb2lab(he); % convert image to L*a*b* colour space
% classify the color in 'a*b*' space using K-Means Clustering
ab = lab_he(:,:,2:3);
ab = im2single(ab);
nColors = 3;
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
subplot(2,4,2),imshow(pixel_labels,[]),title('Image Labeled by Cluster Index');
mask1 = pixel_labels==1;
cluster1 = he .* uint8(mask1);
subplot(2,4,3),imshow(cluster1),title('Objects in Cluster 1'); % white portion
mask2 = pixel_labels==2;
cluster2 = he .* uint8(mask2);
subplot(2,4,4),imshow(cluster2),
title('Objects in Cluster 2'); % pink porion
mask3 = pixel_labels==3;
cluster3 = he .* uint8(mask3);
subplot(2,4,5),imshow(cluster3)
title('Objects in Cluster 3'); % blue portion
L = lab_he(:,:,1);
L_blue = L .* double(mask3);
L_blue = rescale(L_blue);
idx_light_blue = imbinarize(nonzeros(L_blue)); % global thresholding
blue_idx = find(mask3);
mask_dark_blue = mask3;
mask_dark_blue(blue_idx(idx_light_blue)) = 0;
blue_nuclei = he .* uint8(mask_dark_blue);
subplot(2,4,6),imshow(blue_nuclei),
title('Blue Nuclei');
case 3
rgb = imread('pears.png');
I = rgb2gray(rgb);
subplot(5,3,1),imshow(I),
title('Original Image');
gmag = imgradient(I);
subplot(5,3,2),imshow(gmag,[]),
title('Gradient Magnitude')
L = watershed(gmag);
Lrgb = label2rgb(L);
subplot(5,3,3),imshow(Lrgb),
title('Watershed Transform of Gradient Magnitude')
se = strel('disk',20);
Io = imopen(I,se);
subplot(5,3,4),imshow(Io)
title('Opening')
Ie = imerode(I,se);
Iobr = imreconstruct(Ie,I);
subplot(5,3,5),imshow(Iobr)
title('Opening-by-Reconstruction')
Ioc = imclose(Io,se);
subplot(5,3,6),imshow(Ioc)
title('Opening-Closing')
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
subplot(5,3,7),imshow(Iobrcbr)
title('Opening-Closing by Reconstruction')
fgm = imregionalmax(Iobrcbr);
subplot(5,3,8),imshow(fgm)
title('Regional Maxima of Opening-Closing by Reconstruction')
I2 = labeloverlay(I,fgm);
subplot(5,3,9),imshow(I2)
title('Regional Maxima Superimposed on Original Image')
se2 = strel(ones(5,5));
fgm2 = imclose(fgm,se2);
fgm3 = imerode(fgm2,se2);
fgm4 = bwareaopen(fgm3,20);
I3 = labeloverlay(I,fgm4);
subplot(5,3,10),imshow(I3)
title('Modified Regional Maxima Superimposed on Original Image')
bw = imbinarize(Iobrcbr);
subplot(5,3,11),imshow(bw)
title('Thresholded Opening-Closing by Reconstruction')
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
subplot(5,3,12),imshow(bgm)
title('Watershed Ridge Lines)')
gmag2 = imimposemin(gmag, bgm | fgm4);
L = watershed(gmag2);
labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
I4 = labeloverlay(I,labels);
subplot(5,3,13),imshow(I4)
title('Markers and Object Boundaries Superimposed on Original Image')
Lrgb = label2rgb(L,'jet','w','shuffle');
subplot(5,3,14),imshow(Lrgb)
title('Colored Watershed Label Matrix')
figure
subplot(5,3,15),imshow(I)
hold on
himage = imshow(Lrgb);
himage.AlphaData = 0.3;
title('Colored Labels Superimposed Transparently on Original Image')
case 4
I = imread('bag.png');
figure
subplot(6,3,1),imshow(I)
E = entropyfilt(I);
Eim = rescale(E);
figure
subplot(6,3,2),imshow(Eim)
BW1 = imbinarize(Eim, .8);
subplot(6,3,3),imshow(BW1);
figure
subplot(6,3,4),imshow(I)
BWao = bwareaopen(BW1,2000);
subplot(6,3,5),imshow(BWao)
nhood = true(9);
closeBWao = imclose(BWao,nhood);
subplot(6,3,6),imshow(closeBWao)
roughMask = imfill(closeBWao,'holes');
subplot(6,3,7),imshow(roughMask);
figure
subplot(6,3,8),imshow(I)
I2 = I;
I2(roughMask) = 0;
subplot(6,3,9),imshow(I2)
E2 = entropyfilt(I2);
E2im = rescale(E2);
subplot(6,3,10),imshow(E2im)
BW2 = imbinarize(E2im);
subplot(6,3,11),imshow(BW2)
figure,subplot(6,3,12),imshow(I);
mask2 = bwareaopen(BW2,1000);
subplot(6,3,13),imshow(mask2)
texture1 = I;
texture1(~mask2) = 0;
texture2 = I;
texture2(mask2) = 0;
subplot(6,3,14),imshow(texture1)
figure
subplot(6,3,15),imshow(texture2)
boundary = bwperim(mask2);
segmentResults = I;
segmentResults(boundary) = 255;
subplot(6,3,16),imshow(segmentResults)
S = stdfilt(I,nhood);
subplot(6,3,17),imshow(rescale(S))
R = rangefilt(I,ones(5));
subplot(6,3,18),imshow(R)
end
