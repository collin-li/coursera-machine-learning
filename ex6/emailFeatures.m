function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% Return feature vector with 1 if i-th word in dictionary appears in email, 0 otherwise
for i = 1:n
    x(i) = min(sum(i == word_indices),1);
end

end
