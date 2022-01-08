### Title
* Inverse Problems on Evolving Graphs
* Algorithms research
* 

### Graph Example
* Ask about familiarity with graphs
* Explain vertices and edges
* Maybe an animation

### Evolving Graphs
* Idea is: we can add edges at each step
* This leads us to

### Birth/Death Rate
* Give each edge a birth/death rate

### Determining Birth/Death Rate
* One way: "range" between two vertices
* Birth and death rate become functions of the range

### Determining the Range
* This is the inverse problem
* 

### Converting to an Inverse Problem
* Loosen constraints anim

### Methods for solving
* Their solution: eigenanalysis
* Mention my method of gradient descent (very similar to Newton method)

### Testing new methods
* Original paper did not test many
* Mostly just tested a graph derived from range
* Most of the work: testing this on real world graphs
* Their one test was on a small dataset -> let's test larger ones

### Runtime
* Real graphs on the order of 1M vert, 10M edges
* Eigenanalysis is O(n^3)
* My algorithm is #iterations * O(n^2)