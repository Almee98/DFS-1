# Time Complexity : O(n*m) where n is the number of rows and m is the number of columns
# Space Complexity : O(m*n), in case all the oranges are rotten
# Did this code successfully run on Leetcode : Yes
# Any problem you faced while coding this : No

# Approach: BFS
# In this approach, we want to traverse the image in all four directions (up, down, left, right) startng from the given pixel (sr, sc).
# We can traverse in all 4 directions using a queue.
# We will add the starting pixel to the queue and mark it as visited by changing its color to the new color.
# We will then pop the first pixel from the queue and check its 4 neighbors.
# If any of the neighbors is the same color as the starting pixel, we will change its color to the new color and add it to the queue.
# We will repeat this process until the queue is empty.

from collections import deque
class Solution:
    def floodFill(self, image, sr: int, sc: int, color: int):
        # Calculate the number of rows and columns in the image
        m, n = len(image), len(image[0])
        # Initialize a queue to perform BFS
        q = deque()
        # Initialize the directions for the 4 neighbors (up, down, left, right)
        dirs = [[-1,0], [0,1], [1,0], [0,-1]]
        # variable to store the starting color
        start_color = image[sr][sc]
        # if the start color is the same as the color to be filled, we don't need to change the image and we can return the original image
        if start_color == color:
            return image
        # Change the color of the starting pixel to the new color and add it to the queue
        image[sr][sc] = color
        q.append([sr,sc])

        # Perform BFS
        while q:
            # Take a snapshot of the current level. We will process all the pixels at this level in 1 iteration
            cell = q.popleft()
            # Iterate over the 4 neighbors of the cell
            for row, col in dirs:
                r = cell[0] + row
                c = cell[1] + col
                # Check if the neighbor is within bounds and is the same color as the starting pixel
                if r>=0 and c>=0 and r<m and c<n and image[r][c] == start_color:
                    image[r][c] = color
                    q.append([r,c])

        # After processing all the pixels in the queue, we will have the image with the new color
        # Return the updated image
        return image
        

# Time Complexity : O(n*m) where n is the number of rows and m is the number of columns
# Space Complexity : O(m*n), in case all the pixels are the same color
# Approach:
# In this approach, we run a DFS on the image starting from the given pixel (sr, sc), to fill the image with the new color.
class Solution:
    def floodFill(self, image, sr: int, sc: int, color: int):
        m, n = len(image), len(image[0])
        dirs = [[-1,0], [0,1], [1,0], [0,-1]]
        start_color = image[sr][sc]

        # if the start color is the same as the color to be filled, we don't need to change the image and we can return the original image
        if start_color == color:
            return image
        
        def dfs(i, j):
            image[i][j] = color

            for row, col in dirs:
                r = i+row
                c = j+col

                if r>=0 and c>=0 and r<m and c<n:
                    if image[r][c] == start_color:
                        dfs(r, c)

        dfs(sr, sc)
        return image