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
    modelLen = nFeatures*latDim*2;
    initModel = rand(modelLen, 1);
    
    %numerical gradient check
    check_grad(@bprObjective, initModel, 2, 3, 4, iFMat, uiMat);
       
end


function [relRankUij] = relativeRank(U, V, itemFeatMat, userItemMat, u, i, j)
    
    %get item i and j feature, also get their differnce 
    iFeat = itemFeatMat(i,:);
    jFeat = itemFeatMat(j,:);
    featDiff = iFeat - jFeat;
    
    %get set of items rated by user u
    uItems = find(userItemMat(u,:));
    
    %no. of features
    nFeatures = size(itemFeatMat, 2);
    
    %sum features of items rated by user
    if length(uItems) ~= 1
        f_u = sum(itemFeatMat(uItems,:));
    else
        f_u = itemFeatMat(uItems,:);
    end
    r_ui = ((f_u - itemFeatMat(i,:))*U)*(V'*itemFeatMat(i,:)');
    r_uj = (f_u*U)*(V'*itemFeatMat(j,:)');
    relRankUij = r_ui - r_uj;

end


function [Ugrad, Vgrad] = gradientFacRelRank(U, V, itemFeatMat, userItemMat, u, i, j)
    %get item i and j feature, also get their differnce 
    iFeat = itemFeatMat(i,:);
    jFeat = itemFeatMat(j,:);
    featDiff = iFeat - jFeat;
    
    %get set of items rated by user u
    uItems = find(userItemMat(u,:));
    if length(uItems) ~= 1
        f_u = sum(itemFeatMat(uItems,:));
    else
        f_u = itemFeatMat(uItems,:);
    end
    
    Ugrad = (f_u - itemFeatMat(i,:))'*(itemFeatMat(i,:)*V) - f_u'*(itemFeatMat(j,:)*V);
    Vgrad = featDiff'*(f_u*U) - itemFeatMat(i,:)'*(itemFeatMat(i,:)*U);
end


function [loss] = bprObjLoss(sigmaRuij) 
    loss = -log(sigmaRuij);% + factorsReg*(norm(featLatfac, 2)^2);
end


function [f, g] = bprObjective(model, u, i, j, itemFeatMat, userItemMat)
    
    [U, V] = devectorize(model, size(itemFeatMat,2));
    
    relRankUij = relativeRank(U, V, itemFeatMat, userItemMat, u, i, j);
    sigmaRuij = sigmoid(relRankUij);
    [Ugrad, Vgrad] = gradientFacRelRank(U, V, itemFeatMat, userItemMat, u, i, j);
    Ugrad = -(1/(1+exp(relRankUij)))*Ugrad;
    Vgrad = -(1/(1+exp(relRankUij)))*Vgrad;
    
    f = bprObjLoss(sigmaRuij);
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