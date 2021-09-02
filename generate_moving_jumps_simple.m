function [s_modified] = generate_moving_jumps_simple(img)

 %change: dim_pic_x, T, px, last s_modified = ...
    [N_x, N_y] = size(img);

    dim_pic_x = 83;
    dim_pic_y = 83;
    
    %dim_pic_x = 80;
    %dim_pic_y = 80;
    
    T=50; 
    px = 3;
    probab = 0.3;
    
    x_rec(1) = N_x/2;
    y_rec(1) = 1+dim_pic_y;
    
    vector_positions = [x_rec(1); y_rec(1)];
    clear s_modified
    
    for i = 1:T-1
        x_rec(i+1) = x_rec(i);
        y_rec(i+1) = y_rec(i) + px;
        
        col = [x_rec(i+1); y_rec(i+1)];
        vector_positions = [vector_positions,col];
    end

    vector_positions2 = round(vector_positions);

    for i = 1:T
        img = sum(img,3);
        if max(img(:))~=0
            img = img/max(img(:));
        end
        img = img - mean(img(:));
        if max(img(:)) ~=0
            img = img/max(img(:));
        end
        
        s_modified(i,:,:) = img(vector_positions2(1,i)-dim_pic_x:vector_positions2(1,i)+dim_pic_x, vector_positions2(2,i)-dim_pic_y:vector_positions2(2,i)+dim_pic_y);
        s = squeeze(s_modified(i,:,:));
        s = s + 3*randn(2*dim_pic_x+1, 2*dim_pic_y+1);
        %x = rand(dim_pic_x*2+1, dim_pic_y*2+1);
        %x(find(x<probab)) = -1; x(find(x>1-probab)) = 1; x(find((x<probab)&(x>probab))) = 0;
        %s(find(x==-1)) = 0; s(find(x==1)) = 1;
        s_modified(i,:,:) = s;
        %s_modified(i,:,:) = img(vector_positions2(1,i)-160:vector_positions2(1,i)+160, vector_positions2(2,i)-dim_pic_y:vector_positions2(2,i)+dim_pic_y);
    end

 end
