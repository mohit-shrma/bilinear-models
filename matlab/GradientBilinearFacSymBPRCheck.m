function[] = GradientBilinearFacCheck()
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
    modelLen = nFeatures*latDim;
    initModel = rand(modelLen, 1);
    
    %numerical gradient check
    check_grad(@bprObjective, initModel, 2, 5, 4, iFMat, uiMat);
       
end


function [relRankUij] = relativeRank(U, itemFeatMat, userItemMat, u, i, j)
    
    %get item i and j feature, also get their differnce 
    f_i = itemFeatMat(i,:);
    f_j = itemFeatMat(j,:);
    
    %get set of items rated by user u
    uItems = find(userItemMat(u,:));
    
    %sum features of items rated by user
    if length(uItems) ~= 1
        f_u = sum(itemFeatMat(uItems,:));
    else
        f_u = itemFeatMat(uItems,:);
    end
    r_ui = ((f_u - f_i)*(U*U')*f_i');
    r_uj = (f_u*(U*U')*f_j');
    relRankUij = r_ui - r_uj;

end


function [Ugrad] = gradientFacRelRank(U, itemFeatMat, userItemMat, u, i, j)
    %get item i and j feature, also get their differnce 
    f_i = itemFeatMat(i,:);
    f_j = itemFeatMat(j,:);
    
    %get set of items rated by user u
    uItems = find(userItemMat(u,:));
    if length(uItems) ~= 1
        f_u = sum(itemFeatMat(uItems,:));
    else
        f_u = itemFeatMat(uItems,:);
    end
    
    Ugrad = ((f_u - f_i)'*f_i) + (f_i'*(f_u - f_i));
    Ugrad = Ugrad - ((f_u'*f_j) + (f_j'*f_u));
    Ugrad = Ugrad*U;
end


function [loss] = bprObjLoss(sigmaRuij) 
    loss = -log(sigmaRuij);% + factorsReg*(norm(featLatfac, 2)^2);
end


function [f, g] = bprObjective(model, u, i, j, itemFeatMat, userItemMat)
    
    U = devectorize(model, size(itemFeatMat,2));
    
    relRankUij = relativeRank(U, itemFeatMat, userItemMat, u, i, j);
    sigmaRuij = sigmoid(relRankUij);
    Ugrad = gradientFacRelRank(U, itemFeatMat, userItemMat, u, i, j);
    Ugrad = -(1/(1+exp(relRankUij)))*Ugrad;
    
    f = bprObjLoss(sigmaRuij);
    g = vectorize(Ugrad);
end


function [vec] = vectorize(U)
    vec = U(:);
end


function [U] = devectorize(vec, nFeatures)
    latDim = length(vec)/nFeatures;
    U = reshape(vec(1:nFeatures*latDim), nFeatures, latDim);
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