function x = chol2inv(U)
    % Takes a cholesky decomposed matrix and returns the inverse of the
    % original matrix
    
    % Girolami:
    % iU = inv_triu(U);
    
    % --Shakir: instead use
    % This works because mldivide checks for triangular matrices and uses
    % the right algorithm for this inversion.
    iU = U\eye(length(U)); 
    
    x = iU*iU';
end