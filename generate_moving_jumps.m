function [s_modified] = generate_moving_jumps(video)

    load(video)

    N = length(s);
    
    im = s(1).cdata;

    [N_x, N_y] = size(im);
    max_d = sqrt(N_x^2 + N_y^2);

    %dim_pic_x = floor(N_x/6);
    %dim_pic_y = floor(N_y/6);
    dim_pic_x = 83;
    dim_pic_y = 83;
    
    limits = [N_x, N_y, dim_pic_x, dim_pic_y];

    sigma = 1;
    T=500; 

    x_rec(1) = N_x/2;
    y_rec(1) = N_y/2;
    
    theta = 2*pi*rand(1,T-1);
    d = max_d/4 + max_d/8*randn(1,T-1);

    speed_avg = 125;
    speed_std = 0;
    spf = 1/25;

    vector_positions = [x_rec(1); y_rec(1)];
    ind = 1;
    for i = 1:T-1
        %i
        %choose random point
        x_rec(i+1) = dim_pic_x + randi(N_x-2*dim_pic_x);
        y_rec(i+1) = dim_pic_y + randi(N_y-2*dim_pic_y);
    
        target = 0;
        if (x_rec(i) == x_rec(i+1)) && (y_rec(i) == y_rec(i+1))
            target = 1;
        end
    
        x = x_rec(i);
        y = y_rec(i);
        max_d(ind) = norm([x_rec(i+1)-x_rec(i), y_rec(i+1)-y_rec(i)],2);
        theta(ind) = atan2(y_rec(i+1)-y_rec(i), x_rec(i+1)-x_rec(i));
    
        while target == 0
            speed(ind) = speed_avg + speed_std*randn;
            dist(ind) = speed(ind)*spf;
        
            ind = ind+1;

            if dist(ind-1)>max_d(ind-1)
                target = 1;
                col = [x_rec(i+1); y_rec(i+1)];
                vector_positions = [vector_positions,col];
            else
                [x,y] = go_to([x;y], theta(ind-1), dist(ind-1), limits);
                max_d(ind) = norm([y_rec(i+1)-y, x_rec(i+1)-x]);
                theta(ind) = atan2(y_rec(i+1)-y, x_rec(i+1)-x);
                col = [x; y];
                vector_positions = [vector_positions,col];
            end
        
        end
    
    end

    vector_positions2 = round(vector_positions);

    for i = 1:min(T,N)
        img = s(i).cdata;
        img = sum(img,3);
        img = img/max(img(:));
        img = img - mean(img(:));
        img = img/max(img(:));
        
        s_modified(i,:,:) = img(vector_positions2(1,i)-dim_pic_x:vector_positions2(1,i)+dim_pic_x, vector_positions2(2,i)-dim_pic_y:vector_positions2(2,i)+dim_pic_y);
    
    end

    end

function [x2, y2] = go_to(coord, theta, dist, limits)

    x = coord(1);
    y = coord(2);
    
    N_x = limits(1);
    N_y = limits(2);
    dim_pic_x = limits(3);
    dim_pic_y = limits(4);
    
    dx = dist*cos(theta);
    dy = dist*sin(theta);
    
    x2 = x +dx;
    y2 = y +dy;
    
    if x2>N_x-dim_pic_x
       x2 = N_x-dim_pic_x; 
    end
    
    if y2>N_y-dim_pic_y
       y2 = N_y-dim_pic_y; 
    end
    
    if x2<1+dim_pic_x
       x2 = 1+dim_pic_x; 
    end
    
    if y2<1+dim_pic_y
       y2 = 1+dim_pic_y; 
    end
    
end
