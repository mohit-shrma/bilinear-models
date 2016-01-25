function[] = GradientBilinearFacRMSECheck()
     iFMat = [ 1    0    1    0    0;
                1    0    0    1    1;
                1    1    0    0    0;
                0    1    0    1    1;
                0    1    0    0    0;
                1    1    0    0    0;
                1    0    1    0    1;
                1    0    1    1    0;
                1    0    1    0    0;
                1    1    1    0    0];
 
     uiMat = [1    0    0    1    0    0    1    1    0     0;
               1    0    1    0    1    0    1    0    1     1;
               0    1    1    1    0    0    0    1    1     0;
               1    0    1    0    1    0    1    1    0     0;
               1    1    1    0    1    0    0    0    0     1;
               0    0    1    0    1    0    0    0    0     1];
    latDim = 3;
    
    %iFMat = [1 0 1;
    %         0 1 1;
    %         1 1 0];
    % uiMat = [0 1 0;
    %          1 0 1];

    %initialize model
    nFeatures = size(iFMat,2);      
    modelLen = nFeatures*latDim*2;
    initModel = rand(modelLen, 1);
    
    %numerical gradient check
    check_grad(@rmseObjective, initModel, 5, 5, iFMat, uiMat);
       
end


function [Ugrad, Vgrad] = gradientFac(U, V, itemFeatMat, userItemMat, u, i)
    
    %get set of items rated by user u
    uItems = find(userItemMat(u,:));
    if length(uItems) ~= 1
        f_u = sum(itemFeatMat(uItems,:));
    else
        f_u = itemFeatMat(uItems,:);
    end
    
    Ugrad = (f_u - itemFeatMat(i,:))'*(itemFeatMat(i,:)*V);
    Vgrad = itemFeatMat(i,:)'*(f_u*U) - itemFeatMat(i,:)'*(itemFeatMat(i,:)*U);
end


function [rating] = estRating(U, V, u , i, itemFeatMat, userItemMat)
    
    %get set of items rated by user u
    uItems = find(userItemMat(u,:));
    if length(uItems) ~= 1
        f_u = sum(itemFeatMat(uItems,:));
    else
        f_u = itemFeatMat(uItems,:);
    end
    rating = (f_u - itemFeatMat(i,:))*U*V'*itemFeatMat(i,:)';
end


function [f, g] = rmseObjective(model, u, i, itemFeatMat, userItemMat)
    
    [U, V] = devectorize(model, size(itemFeatMat,2));
    r_ui_est = estRating(U, V, u , i, itemFeatMat, userItemMat);
    r_ui = userItemMat(u,i);
    [Ugrad, Vgrad] = gradientFac(U, V, itemFeatMat, userItemMat, u, i);
    Ugrad = 2*(r_ui_est - r_ui)*Ugrad;
    Vgrad = 2*(r_ui_est - r_ui)*Vgrad;
    
    %se loss
    f = (r_ui_est - r_ui)*(r_ui_est - r_ui);
    g = vectorize(Ugrad, Vgrad);
end


function [vec] = vectorize(U, V)
    vec = [U(:); V(:)];
end


function [U, V] = devectorize(vec, nFeatures)
    latDim = length(vec)/(2*nFeatures);
    U = reshape(vec(1:nFeatures*latDim), nFeatures, latDim);
    V = reshape(vec(nFeatures*latDim+1: length(vec)), nFeatures, latDim);
    %fprintf('\nlatDim = %f', latDim);
end


function[sigmaRuij] = sigmoid(ruij)
    sigmaRuij = 1.0/(1 + exp(-ruij));
end

function []  = check_grad(f, x0, varargin)

delta = rand(size(x0));
delta = delta ./ norm(delta);
epsilon = 10.^[-7:-1];

[f0, df0] = feval(f, x0, varargin{:});

    for i = 1:length(epsilon)
        [f_left] = feval(f, x0-epsilon(i)*delta, varargin{:});
        [f_right] = feval(f, x0+epsilon(i)*delta, varargin{:});
        ys(i) = (f_right - f_left) / 2;
        ys_hat(i) = df0' * epsilon(i)*delta;    
        fprintf('epsilon: %d , gradient: %d \n', epsilon(i), ys(i) / ys_hat(i));
    end   
end