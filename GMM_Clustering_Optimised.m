clear;
clc;
close all;

rng(10922540,'twister');

trueK = 5;
pointsPerCluster = 1000;
N = trueK * pointsPerCluster;

mu = [-7 0; -2 6; 4 5; 7 -2; 0 -5];
sigma = 1.2;

X = zeros(N,2);
trueLabels = zeros(N,1);

for c = 1:trueK
    startIdx = (c-1)*pointsPerCluster + 1;
    endIdx = c*pointsPerCluster;
    X(startIdx:endIdx,:) = mu(c,:) + sigma*randn(pointsPerCluster,2);
    trueLabels(startIdx:endIdx) = c;
end

disp('Total observations:');
disp(N);

figure;
scatter(X(:,1),X(:,2),10,trueLabels,'filled');
title('Original Dataset: Five Gaussian Clusters');
xlabel('Feature X');
ylabel('Feature Y');
grid on;

K_values = 2:8;
maxIter = 100;
numRestarts = 1;
regularisation = 1e-5;
tolerance = 1e-4;

BIC = zeros(length(K_values),1);
logLikelihoods = zeros(length(K_values),1);
GMM_WCSS = zeros(length(K_values),1);

labelsGMM5 = zeros(N,1);
bestGMM5WCSS = 0;
bestGMM5LL = 0;

rng(10922541,'twister');

globalCov = cov(X) + regularisation*eye(2);

for kIndex = 1:length(K_values)

    K = K_values(kIndex);
    disp(['Testing GMM with K = ',num2str(K)]);

    bestLogLikelihood = -1e100;
    bestResponsibilities = [];

    for restart = 1:numRestarts

        selected = randperm(N,K);
        means = X(selected,:);
        covariances = zeros(2,2,K);

        for j = 1:K
            covariances(:,:,j) = globalCov;
        end

        weights = ones(1,K)/K;
        oldLogLikelihood = -1e100;

        for iter = 1:maxIter

            weightedPDF = zeros(N,K);

            for j = 1:K
                difference = X - means(j,:);
                currentCov = covariances(:,:,j);
                determinant = det(currentCov);

                if determinant <= 0
                    currentCov = currentCov + regularisation*eye(2);
                    covariances(:,:,j) = currentCov;
                    determinant = det(currentCov);
                end

                inverseCov = inv(currentCov);
                mahalanobis = sum((difference*inverseCov).*difference,2);
                gaussianPDF = exp(-0.5*mahalanobis)/(2*pi*sqrt(determinant));
                gaussianPDF = max(gaussianPDF,realmin);
                weightedPDF(:,j) = weights(j)*gaussianPDF;
            end

            totalPDF = sum(weightedPDF,2);
            totalPDF = max(totalPDF,realmin);
            responsibilities = weightedPDF./totalPDF;
            logLikelihood = sum(log(totalPDF));

            Nk = sum(responsibilities,1);

            for j = 1:K
                if Nk(j) > 1e-8
                    means(j,:) = (responsibilities(:,j)'*X)/Nk(j);
                    difference = X - means(j,:);
                    weightedDifference = difference .* responsibilities(:,j);
                    currentCov = (difference'*weightedDifference)/Nk(j);
                    covariances(:,:,j) = currentCov + regularisation*eye(2);
                    weights(j) = Nk(j)/N;
                else
                    randomPoint = randi(N);
                    means(j,:) = X(randomPoint,:);
                    covariances(:,:,j) = globalCov;
                    weights(j) = 1/N;
                end
            end

            weights = weights/sum(weights);

            if abs(logLikelihood-oldLogLikelihood) < tolerance
                break;
            end

            oldLogLikelihood = logLikelihood;
        end

        weightedPDF = zeros(N,K);

        for j = 1:K
            difference = X - means(j,:);
            currentCov = covariances(:,:,j);
            determinant = max(det(currentCov),realmin);
            inverseCov = inv(currentCov);
            mahalanobis = sum((difference*inverseCov).*difference,2);
            gaussianPDF = exp(-0.5*mahalanobis)/(2*pi*sqrt(determinant));
            gaussianPDF = max(gaussianPDF,realmin);
            weightedPDF(:,j) = weights(j)*gaussianPDF;
        end

        totalPDF = max(sum(weightedPDF,2),realmin);
        responsibilities = weightedPDF./totalPDF;
        finalLogLikelihood = sum(log(totalPDF));

        if finalLogLikelihood > bestLogLikelihood
            bestLogLikelihood = finalLogLikelihood;
            bestResponsibilities = responsibilities;
        end
    end

    numberParameters = 6*K - 1;
    BIC(kIndex) = -2*bestLogLikelihood + numberParameters*log(N);
    logLikelihoods(kIndex) = bestLogLikelihood;

    [~,predictedLabels] = max(bestResponsibilities,[],2);
    currentWCSS = 0;

    for j = 1:K
        clusterPoints = X(predictedLabels==j,:);
        if ~isempty(clusterPoints)
            clusterMean = mean(clusterPoints,1);
            currentWCSS = currentWCSS + sum(sum((clusterPoints-clusterMean).^2));
        end
    end

    GMM_WCSS(kIndex) = currentWCSS;

    if K == 5
        labelsGMM5 = predictedLabels;
        bestGMM5WCSS = currentWCSS;
        bestGMM5LL = bestLogLikelihood;
    end
end

[~,bestIndex] = min(BIC);
suggestedK = K_values(bestIndex);

fprintf('\nSuggested number of GMM components: %d\n',suggestedK);
fprintf('Known number of clusters: %d\n',trueK);

figure;
plot(K_values,BIC,'-o','LineWidth',2);
title('GMM Model Selection Using BIC');
xlabel('Number of Mixture Components (K)');
ylabel('Bayesian Information Criterion (BIC)');
grid on;

figure;
scatter(X(:,1),X(:,2),10,labelsGMM5,'filled');
hold on;

for j = 1:5
    clusterPoints = X(labelsGMM5==j,:);
    if ~isempty(clusterPoints)
        componentMean = mean(clusterPoints,1);
        plot(componentMean(1),componentMean(2),'kx','MarkerSize',12,'LineWidth',3);
    end
end

title('GMM Clustering with K = 5');
xlabel('Feature X');
ylabel('Feature Y');
grid on;
hold off;

contingency = zeros(trueK,trueK);

for i = 1:N
    actual = trueLabels(i);
    predicted = labelsGMM5(i);
    contingency(actual,predicted) = contingency(actual,predicted) + 1;
end

n = sum(contingency(:));
index = sum(sum(contingency.*(contingency-1)/2));
rowSum = sum(contingency,2);
colSum = sum(contingency,1);
sumRows = sum(rowSum.*(rowSum-1)/2);
sumCols = sum(colSum.*(colSum-1)/2);
totalPairs = n*(n-1)/2;
expectedIndex = (sumRows*sumCols)/totalPairs;
maxIndex = 0.5*(sumRows+sumCols);

if maxIndex == expectedIndex
    ARI_GMM = 1;
else
    ARI_GMM = (index-expectedIndex)/(maxIndex-expectedIndex);
end

fprintf('\n========== GMM FINAL RESULTS ==========\n');
fprintf('Total observations: %d\n',N);
fprintf('Known number of clusters: %d\n',trueK);
fprintf('Suggested number of GMM components: %d\n',suggestedK);
fprintf('GMM log-likelihood for K=5: %.4f\n',bestGMM5LL);
fprintf('GMM WCSS for K=5: %.4f\n',bestGMM5WCSS);
fprintf('GMM Adjusted Rand Index: %.4f\n',ARI_GMM);

fprintf('\nBIC values:\n');
for i = 1:length(K_values)
    fprintf('K = %d, BIC = %.4f\n',K_values(i),BIC(i));
end

fprintf('\nGMM WCSS values:\n');
for i = 1:length(K_values)
    fprintf('K = %d, WCSS = %.4f\n',K_values(i),GMM_WCSS(i));
end

KMeans_ARI = 0.9915;
KMeans_WCSS = 14810.2880;

fprintf('\n========== MODEL COMPARISON ==========\n');
fprintf('K-Means ARI: %.4f\n',KMeans_ARI);
fprintf('GMM ARI: %.4f\n',ARI_GMM);
fprintf('K-Means WCSS: %.4f\n',KMeans_WCSS);
fprintf('GMM WCSS: %.4f\n',bestGMM5WCSS);
fprintf('======================================\n');
