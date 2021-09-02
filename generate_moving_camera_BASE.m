%change: save_dir
save_dir = '/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_valid_noiseG';
if ~exist(save_dir,'dir')
   mkdir(save_dir);
end
%img_dir = '/home/dvoina/ramsmatlabprogram/BSR_2/BSDS500/data/images/train/im_dir';
img_dir = '/home/dvoina/ramsmatlabprogram/BSR_2/BSDS500/data/images/test/';

img_list = dir(img_dir);
img_files = {img_list.name}';

L = length(img_files);

for i = 4:L-1
    i
    a = img_files(i);
    fullfile(strcat(img_dir, '/', a{1}))
    I = imread(fullfile(strcat(img_dir, '/', a{1})));
    img = I;
    
    img = sum(img,3);     %avg over color axis
    if max(img(:)) ~=0
      img = img/max(img(:));
    end
    img = img - mean(img(:));
    if max(img(:)) ~=0
        img = img/max(img(:));
    end
       
    s_modified = generate_moving_jumps_simple(img);
    save(fullfile(save_dir,strcat(a{1}(1:end-4), '.mat')), 's_modified')
end
