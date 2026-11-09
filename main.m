clear;
clc;
close all;

% Dataset

rng(10922540,'twister');

trueK = 5;
pointsPerCluster = 1000;
N = trueK * pointsPerCluster;

% Assumptions:
% Five 2D Gaussian clusters
% Two independent features with equal variance
% 1000 observations per cluster
% Standard deviation = 1.2

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

%Original dataset

figure;
scatter(X(:,1),X(:,2),10,trueLabels,'filled');
title('Original Dataset: Five Gaussian Clusters');
xlabel('Feature X');
ylabel('Feature Y');
grid on;



%K-MEANS CLUSTERING WITH K = 5


rng(10922541,'twister');

K = 5;
maxIter = 300;
numRestarts = 10;

bestWCSS = 1e100;

disp('Starting K-Means with K = 5');

for restart = 1:numRestarts

    centroids = zeros(K,2);

    first = randi(N);
    centroids(1,:) = X(first,:);

    minDist = sum((X-centroids(1,:)).^2,2);

    for j = 2:K

        if sum(minDist) == 0
            next = randi(N);
        else
            probabilities = minDist/sum(minDist);
            cumulativeProb = cumsum(probabilities);
            r = rand();

            next = find(cumulativeProb >= r,1);

            if isempty(next)
                next = randi(N);
            end
        end

        centroids(j,:) = X(next,:);

        dist = sum((X-centroids(j,:)).^2,2);
        minDist = min(minDist,dist);

    end

    %Lloyd's algorithm
    for iter = 1:maxIter

        oldCentroids = centroids;

        distances = sum(X.^2,2) + sum(centroids.^2,2)' - 2*X*centroids';
        distances = max(distances,0);

        [minDistances,labels] = min(distances,[],2);

        for j = 1:K

            clusterPoints = X(labels==j,:);

            if ~isempty(clusterPoints)
                centroids(j,:) = mean(clusterPoints,1);
            else
                [~,farthest] = max(minDistances);
                centroids(j,:) = X(farthest,:);
            end

        end

        movement = sqrt(sum((centroids-oldCentroids).^2,2));

        if max(movement) < 1e-5
            break;
        end

    end

    distances = sum(X.^2,2) + sum(centroids.^2,2)' - 2*X*centroids';
    distances = max(distances,0);

    [minDistances,labels] = min(distances,[],2);

    currentWCSS = sum(minDistances);

    if currentWCSS < bestWCSS

        bestWCSS = currentWCSS;
        bestLabels = labels;
        bestCentroids = centroids;

    end

end

labels5 = bestLabels;
centroids5 = bestCentroids;
wcss5 = bestWCSS;

disp('K-Means calculation completed');


%Clustering results

figure;
scatter(X(:,1),X(:,2),10,labels5,'filled');
hold on;
plot(centroids5(:,1),centroids5(:,2),'kx','MarkerSize',12,'LineWidth',3);
title('K-Means Clustering with K = 5');
xlabel('Feature X');
ylabel('Feature Y');
grid on;
hold off;

% Test Values

K_values = 2:8;
WCSS = zeros(length(K_values),1);

for i = 1:length(K_values)

    K = K_values(i);

    rng(10922542 + K,'twister');

    bestWCSS = 1e100;

    disp(['Testing K = ',num2str(K)]);

    for restart = 1:5

        centroids = zeros(K,2);

        first = randi(N);
        centroids(1,:) = X(first,:);

        minDist = sum((X-centroids(1,:)).^2,2);

        for j = 2:K

            if sum(minDist) == 0
                next = randi(N);
            else
                probabilities = minDist/sum(minDist);
                cumulativeProb = cumsum(probabilities);
                r = rand();

                next = find(cumulativeProb >= r,1);

                if isempty(next)
                    next = randi(N);
                end
            end

            centroids(j,:) = X(next,:);

            dist = sum((X-centroids(j,:)).^2,2);
            minDist = min(minDist,dist);

        end

        % Lloyd's algorithm
        for iter = 1:maxIter

            oldCentroids = centroids;

            distances = sum(X.^2,2) + sum(centroids.^2,2)' - 2*X*centroids';
            distances = max(distances,0);

            [minDistances,labels] = min(distances,[],2);

            for j = 1:K

                clusterPoints = X(labels==j,:);

                if ~isempty(clusterPoints)
                    centroids(j,:) = mean(clusterPoints,1);
                else
                    [~,farthest] = max(minDistances);
                    centroids(j,:) = X(farthest,:);
                end

            end

            movement = sqrt(sum((centroids-oldCentroids).^2,2));

            if max(movement) < 1e-5
                break;
            end

        end

        distances = sum(X.^2,2) + sum(centroids.^2,2)' - 2*X*centroids';
        distances = max(distances,0);

        [minDistances,labels] = min(distances,[],2);

        currentWCSS = sum(minDistances);

        if currentWCSS < bestWCSS
            bestWCSS = currentWCSS;
        end

    end

    WCSS(i) = bestWCSS;

end


% Elbow method

figure;
plot(K_values,WCSS,'-o','LineWidth',2);
title('Elbow Method for Selecting K');
xlabel('Number of Clusters (K)');
ylabel('Within-Cluster Sum of Squares (WCSS)');
grid on;


% Optimal K using elbow


x1 = K_values(1);
y1 = WCSS(1);

x2 = K_values(end);
y2 = WCSS(end);

elbowDistances = zeros(length(K_values),1);

for i = 2:length(K_values)-1

    x = K_values(i);
    y = WCSS(i);

    elbowDistances(i) = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) ...
        / sqrt((y2-y1)^2 + (x2-x1)^2);

end

[~,elbowIndex] = max(elbowDistances);
suggestedK = K_values(elbowIndex);

fprintf('\nSuggested K from elbow method: %d\n',suggestedK);
fprintf('Known number of clusters: %d\n',trueK);


% Performance evaluation

contingency = zeros(trueK,trueK);

for i = 1:N

    actual = trueLabels(i);
    predicted = labels5(i);

    contingency(actual,predicted) = contingency(actual,predicted) + 1;

end

% Calculate adjusted Rand Index
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
    ARI = 1;
else
    ARI = (index-expectedIndex)/(maxIndex-expectedIndex);
end


%Display 

fprintf('\n========== FINAL RESULTS ==========\n');

fprintf('Total observations: %d\n',N);
fprintf('Points per cluster: %d\n',pointsPerCluster);

fprintf('Known number of clusters: %d\n',trueK);
fprintf('Suggested number of clusters: %d\n',suggestedK);

fprintf('WCSS for K = 5: %.4f\n',wcss5);

fprintf('Adjusted Rand Index: %.4f\n',ARI);

fprintf('\nWCSS for each tested K:\n');

for i = 1:length(K_values)

    fprintf('K = %d, WCSS = %.4f\n',K_values(i),WCSS(i));

end

fprintf('===================================\n');