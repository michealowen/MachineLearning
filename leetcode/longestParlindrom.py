class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        #首先将s逆置
        s_ = s[::-1]
        #构造矩阵
        l = len(s)
        maxlength = 0
        beginIndex = 0
        matrix = [[0 for i in range(l)] for i in range(l)] 
        for i in range(l):
            for j in range(l):
                if s[i] == s_[j]:
                    if i == 0 or j == 0:
                        matrix[i][j] = 1
                    else:
                        matrix[i][j] = matrix[i-1][j-1]+1
                if matrix[i][j] > maxlength:
                    maxlength = matrix[i][j]
                    beginIndex = i - maxlength + 1
        
        return s[beginIndex:beginIndex+maxlength]