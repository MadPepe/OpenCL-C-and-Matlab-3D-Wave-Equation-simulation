clear;
fileID = fopen('daneX.txt','r');

formatSpec = '%f %f %f %f %f';
sizeA = [5 inf];

A = fscanf(fileID, formatSpec, sizeA);
A = A';
Uz = A(11:20:end,5);
Uy = [];
Ux = [];

for i = 1:(numel(A(:,5)))
    buffery = A(i,3);
    bufferx = A(i,2);
    if(buffery == 10)
        Uy = [Uy (A(i,5))];
    end
    if(bufferx == 10)
        Ux = [Ux (A(i,5))];
    end
end

Uy = Uy';
Ux = Ux';

fclose(fileID);
[Xz,Yz] = meshgrid(-9:1:10,-9:1:10);
[Xy,Zy] = meshgrid(-9:1:10,-9:1:10);
[Yx,Zx] = meshgrid(-9:1:10,-9:1:10);


iterations = numel(Uz)/(20*20);

for j = 1:3
    
    for i = 1:iterations
        if(i == 1)
            fin = i*(20*20);
            start = i;
        else
            fin = i*(20*20);
            start = (i*(20*20)-(20*20))+1;
        end
        Bz = Uz(start:fin);
        By = Uy(start:fin);
        Bx = Ux(start:fin);
        Hz = reshape(Bz, [20,20]);
        Hy = reshape(By, [20,20]);
        Hx = reshape(Bx, [20,20]);
        surf(Xz,Yz,Hz)
        hold on
        surf(Xy,Hy,Zy)
        hold on
        surf(Hx,Yx,Zx)
        xlim([-9 10])
        ylim([-9 10])
        zlim([-9 10])
        hold off;
        pause(0.005);
        %pause(1.5);
    end
end



