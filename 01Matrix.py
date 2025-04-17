from collections import deque

# Time Complexity: O(m*n), because we are processing each cell exactly once
# Space Complexity: O(m*n), because we are using a queue to store the cells
# Approcah:
# In this approcah, we will use BFS to traverse the grid.
# We will maintain a queue to keep track of the cells to be processed.
# We will also maintain a distance variable to keep track of the distance from the nearest 0.
# We want to start by adding all the 0s to the queue, so we know that we will only be processing 1s in the next level.
# To avoid, processing the same cells again, we initially multiply all the 1s by -1, and then replace them by the distance.
class Solution:
    def updateMatrix(self, mat):
        # Calculate the number of rows and columns in the grid
        m, n = len(mat), len(mat[0])
        # directions for the 4 neighbors (up, down, left, right)
        dirs = [[-1,0],[1,0],[0,1],[0,-1]]
        # Initialize a queue to perform BFS
        q = deque()
        # Initialize distance variable
        dist = 0

        # Iterate over the grid to find all the 0s and 1s. If the cell is 0, we will add it to the queue
        # If the cell is 1, we will multiply it by -1 to mark it as visited
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    q.append([i,j])
                if mat[i][j] == 1:
                    mat[i][j] *= -1

        # Perform BFS
        while q:
            # Take a snapshot of the current level. We will process all the cells at this level in 1 iteration
            size = len(q)
            # Increment the distance
            dist += 1
            # Process all the cells at this level
            for i in range(size):
                # Pop the first cell from the queue, and get it's row and column
                cell = q.popleft()
                # Iterate over the 4 neighbors of the cell
                for row, col in dirs:
                    r = cell[0] + row
                    c = cell[1] + col
                    # Check if the neighbor is within bounds and is not visited before
                    if r>=0 and c>=0 and r<m and c<n and mat[r][c] == -1:
                        # If the neighbor is not visited before, we will add it to the queue and update its value with the distance
                        q.append([r,c])
                        mat[r][c] = dist
        # After processing all the cells in the queue, we will have the distance from the nearest 0 for all the cells
        return mat
    

# This approcah is the same as previous one, but instead of multiplying the 1s by -1, we take an offset to the distance. This will help us avoid visiting the same cell again.
# Time Complexity: O(m*n), because we are processing each cell exactly once
# Space Complexity: O(m*n), because we are using a queue to store the cells
class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])
        dirs = [[-1,0],[1,0],[0,1],[0,-1]]
        q = deque()
        dist = 2

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    q.append([i,j])

        while q:
            size = len(q)
            dist += 1
            for i in range(size):
                cell = q.popleft()
                for row, col in dirs:
                    r = cell[0] + row
                    c = cell[1] + col
                    if r>=0 and c>=0 and r<m and c<n and mat[r][c] == 1:
                        q.append([r,c])
                        mat[r][c] = dist
            

        for i in range(m):
            for j in range(n):
                if mat[i][j] != 0:
                    mat[i][j] -= 2

        return mat

# This solution is similar to the preious one, but instead of taking a snapshot of the current level, we will process all the cells at this level in 1 iteration.
# We are sure that we will only be moving to the next level, and not processing the same cells again.
# So we can simply add 1 to the distance of the previous cell and assign it to the current cell.
class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])
        dirs = [[-1,0],[1,0],[0,1],[0,-1]]
        q = deque()
        dist = 0

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    q.append([i,j])
                if mat[i][j] == 1:
                    mat[i][j] *= -1

        while q:
            cell = q.popleft()
            for row, col in dirs:
                r = cell[0] + row
                c = cell[1] + col
                if r>=0 and c>=0 and r<m and c<n and mat[r][c] == -1:
                    q.append([r,c])
                    mat[r][c] = mat[cell[0]][cell[1]] + 1
            
        return mat



# BFS Approach:
# Time Complexity : O(n^2 * m^2) because for each 1 in the grid, we will run a BFS to find the nearest 0
# Space Complexity : O(n * m) because we will maintain a queue to keep track of the cells to be processed

# In this approach, we will use BFS to traverse the grid.
# We run a BFS for all the 1s in the grid and find the distance to the nearest 0.
# The dfs() function will have a queue to keep track of the cells to be processed and the time taken
# We will also need to maintain a visited set to keep track of the cells that have already been processed, in order to avoid visiting them again.
class Solution:
    def updateMatrix(self, mat):
        # Calculate the number of rows and columns in the grid
        m, n = len(mat), len(mat[0])
        # Initialize a queue to perform BFS
        dirs = [[-1,0],[1,0],[0,1],[0,-1]]

        # BFS function to find the distance to the nearest 0
        def bfs(i, j, time):
            # Initialize a queue and a visited set and add the current cell to the queue and visited set
            q = deque()
            visited = set()
            q.append([i,j])
            visited.add((i,j))
            while q:
                # Take a snapshot of the current level. We will process all the cells at this level in 1 iteration
                # Increment the time
                time += 1
                size = len(q)
                # Process all the cells at this level
                for k in range(size):
                    # Pop the first cell from the queue, and get it's row and column
                    cell = q.popleft()
                    # Iterate over the 4 neighbors of the cell
                    for row, col in dirs:
                        r = cell[0]+row
                        c = cell[1]+col
                        # Check if the neighbor is within bounds
                        if r>=0 and c>=0 and r<m and c<n:
                            # Check if the neighbor is a 0
                            # If the neighbor is a 0, we have found the nearest 0, so return the time
                            if mat[r][c] == 0:
                                return time
                            # If the neighbor is a 1 and has not been visited yet, add it to the queue and visited set
                            if (r,c) not in visited:
                                q.append([r,c])
                                visited.add((r,c))

        # Iterate over the grid to find all the 1s and run a BFS for each 1, and update the value of the cell to the distance to the nearest 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    mat[i][j] = bfs(i, j, 0)

        # Return the updated grid
        return mat
    
    
# Time Complexity : O(m*n) where m is the number of rows and n is the number of columns
# Space Complexity : O(m+n), because we reuse the result for 1s from previous DFS calls
# In this approach, we will use DFS to traverse the grid.
# We will maintain a result matrix to keep track of the distance to the nearest 0.
# Whenever we encounter a 1, we want to perform a DFS to find the distance to the nearest 0.
# If we reach a 0 in any of the 4 directions, we will return 1.
# Otherwise, we will caryy out the recursion in bottom and right directions and return the minimum distance from the current cell to the nearest 0.
# We will not go to top and left because in some cases, we will be coming from those directions and going back will result in a cycle.

class Solution:
    def updateMatrix(self, mat):
        # Calculate the number of rows and columns in the grid
        m, n = len(mat), len(mat[0])
        # directions for the 4 neighbors (up, down, left, right)
        dirs = [[-1,0],[1,0],[0,1],[0,-1]]
        # Initialize a result matrix with 0s to keep track of the distance to the nearest 0
        res = [[0 for _ in range(n)] for _ in range(m)]

        # DFS function to find the distance to the nearest 0
        def dfs(i, j):
            # If a cell in the 4 neighbors is a 0, we have found the nearest 0, so return 1
            for row, col in dirs:
                r = i+row
                c = j+col
                if r>=0 and r<m and c>=0 and c<n:
                    if mat[r][c] == 0:
                        return 1
                    
            # Otherwise, to calculate the distance to the nearest 0, we need to get the minimum from all the 4 neighbors and add 1 to it
            # We will initialize the distance to a large number, so that we can get the minimum from all the 4 neighbors
            top = 9999
            left = 9999
            bottom = 9999
            right = 9999

            # If the top value in the result matrix is not 0, it means that we already have the distance to the nearest 0 from the top cell
            # So we can replace the top value with the distance from the top cell
            if i>0 and res[i-1][j] != 0:
                top = res[i-1][j]
            
            # Similarly, if the left value in the result matrix is not 0, it means that we already have the distance to the nearest 0 from the left cell
            # So we can replace the left value with the distance from the left cell
            if j>0 and res[i][j-1] != 0:
                left = res[i][j-1]

            # Now, if bottom value in the result matrix is 0, it means that we have not yet calculated the distance to the nearest 0 from the bottom cell
            if i < m-1:
                # So we will call dfs on the bottom cell and get the distance to the nearest 0 and store the result in the result matrix
                if res[i+1][j] == 0:
                    res[i+1][j] = dfs(i+1, j)
                # Now we can replace the bottom value with the distance from the bottom cell
                bottom = res[i+1][j]

            # Similarly, if the right value in the result matrix is 0, it means that we have not yet calculated the distance to the nearest 0 from the right cell
            if j < n-1:
                # So we will call dfs on the right cell and get the distance to the nearest 0 and store the result in the result matrix
                if res[i][j+1] == 0:
                    res[i][j+1] = dfs(i, j+1)
                # Now we can replace the right value with the distance from the right cell
                right = res[i][j+1]

            # Now we can return the minimum distance from the current cell to the nearest 0
            # We will add 1 to the minimum distance from all the 4 neighbors
            return 1 + min(top, left, bottom, right)
        
        # Iterate over the grid to find all the 1s and run a DFS for each 1, and update the value of the cell to the distance to the nearest 0 in the result matrix.
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    res[i][j] = dfs(i, j)

        # Return the updated result matrix
        return res

# DFS Approcah:
# Time Complexity : O(n^2 * m^2) because for each 0 in the grid, we will run a DFS to find the distance to 1s
# Space Complexity : O(m * n) recursion stack
# This approcah is similar to the DFS approcah for rotten oranges.
# For each 0 in the grid, we will calculate the distance to all the 1s in the grid.
# We will update the distance of a 1 if a we can explore it in a shorter distance.
# This approcah will result in 'time limit exceeded' for large grids, because we are running a DFS for each 0 in the grid.
class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])
        dirs = [[-1,0],[1,0],[0,1],[0,-1]]
        
        def dfs(i, j, dist):

            mat[i][j] = dist

            for row, col in dirs:
                r = i + row
                c = j + col
                if r>=0 and c>=0 and r<m and c<n and (mat[r][c] == 1 or mat[r][c] > dist):
                    dfs(r, c, dist+1)

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    # sending 2 as an offset
                    dfs(i, j, 2)

        # subtracting the offset from the final result
        for i in range(m):
            for j in range(n):
                mat[i][j] -= 2

        return mat
    

# Tabulation Approach:
# Time Complexity : O(2*m*n) because we are processing each cell exactly twice
# Space Complexity : O(1), not counting the resultant matrix

# In this approach, the intuition is that for each 1 in the grid, it's distance to the nearest 0 will be 1 + the minimum distance from the top, left, bottom and right cells.
# For all 1s in the input matrix, we will initialize the resultant matrix with a large number.
# Now, we will iterate over the grid from top to bottom and left to right, and for each 1, we will check the top and left cells and update the distance in the resultant matrix.
# Then we will iterate over the grid from bottom to top and right to left, and for each 1, we will check the bottom and right cells and update the distance in the resultant matrix again.
# This will ensure that we are getting the minimum distance from all the 4 neighbors.
class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])
        res = [[9999 for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    res[i][j] = 0

        # Tabulataion approach
        # First pass - top to bottom, left to right
        for i in range(m):
            for j in range(n):
                # If the cell is 0, we don't want to update it.
                # We will only update the cells that are 1s
                if mat[i][j] == 1:
                    # If the top and left cells exist, we will take the minimum of the top and left cells and add 1 to it
                    if i>0 and j>0:
                        res[i][j] = 1 + min(res[i-1][j], res[i][j-1])
                    # If only the top cell exists, we will take the top cell and add 1 to it
                    elif i>0:
                        res[i][j] = 1 + res[i-1][j]
                    # If only the left cell exists, we will take the left cell and add 1 to it
                    elif j>0:
                        res[i][j] = 1 + res[i][j-1]
        

        # Second pass - bottom to top, right to left
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                # We will only update the cells that are 1s
                if mat[i][j] == 1:
                    # If the bottom and right cells exist, we will take the minimum of the bottom and right cells and add 1 to it
                    # And get the minimum of that, and alrady calculated minimum in that cell, and update the value accordingly.
                    if i<m-1 and j<n-1:
                        res[i][j] = min(res[i][j], min(res[i+1][j], res[i][j+1])+1)
                    # If only the bottom cell exists, we will take the bottom cell and add 1 to it, and compare it with already existing value in that cell
                    elif i<m-1:
                        res[i][j] = min(res[i][j], res[i+1][j]+1)
                    # If only the right cell exists, we will take the right cell and add 1 to it, and compare it with already existing value in that cell
                    elif j<n-1:
                        res[i][j] = min(res[i][j], res[i][j+1]+1)

        # Return the updated result matrix
        return res