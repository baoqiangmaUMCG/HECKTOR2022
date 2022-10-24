#' full.combinations
#'
#' Returns all combinations of elements that do not conflict eachother and that
#' cannot contain more elements without conflict.
#'
#' The algorithm searches recursively, starting from an empty combination (all
#' elements not selected) with a pointer to the first element. If the pointed
#' element has a conflict with a selected elements to the left, then the element
#' is not inserted and the pointer is advanced. Else, if the pointed element has
#' no possible conflict with any element to the right (with higher index), then
#' the element is inserted and the pointer is advanced to the right (index
#' increased by one). Otherwise, the combination is split: one combination does
#' not contain the pointed element and one does. In both cases the pointer is
#' advanced. The combination is full if the pointer is advanced beyond the last
#' element. This procedure may produce combinations that are subsets of other
#' combinations. These are removed at the end.
#'
#' @param conflict.matrix square logical matrix of which the element [i, j]
#'   indicates if element i conflicts with element j. The conflict matrix is
#'   coerced into logical and made symmetric by logical OR operation with its own
#'   transpose, such that one-sided conflicts become mutual. The diagonal
#'   elements are ignored.
#'
#' @return list of logical vectors, each indicating the elements of one
#'   combination.
#' @export
#'
#' @examples
#' n <- 5
#' conflict.matrix <- matrix(runif(n^2) < 0.3, nrow = n)
#' full.combinations(conflict.matrix)
#'
full.combinations <- function(conflict.matrix,
                              pointer = 1,
                              status = rep(FALSE, nrow(conflict.matrix)))
  {
  stopifnot(is.matrix(conflict.matrix))
  n <- ncol(conflict.matrix)
  stopifnot(nrow(conflict.matrix) == n)
  if (mode(conflict.matrix) != "logical") conflict.matrix <- conflict.matrix != 0
  if (!isSymmetric.matrix(conflict.matrix)) conflict.matrix <- conflict.matrix | t(conflict.matrix)

  if (pointer > n) {
    # pointer is advanced beyond the last element
    result <- list(status)
  } else if (pointer > 1 && any(conflict.matrix[pointer, 1:(pointer-1)] & status[1:(pointer-1)])) {
    # do not select pointed element because of conflict to the left
    result <- full.combinations(conflict.matrix, pointer = pointer + 1, status = status)
  } else if (pointer == n || !any(conflict.matrix[pointer, (pointer+1):n])) {
    # select pointed element because of no possible conflict to the right
    status[pointer] <- TRUE
    result <- full.combinations(conflict.matrix, pointer = pointer + 1, status = status)
  } else {
    # split combination
    one <- full.combinations(conflict.matrix, pointer = pointer + 1, status = status)
    status[pointer] <- TRUE
    two <- full.combinations(conflict.matrix, pointer = pointer + 1, status = status)
    result <- append(one, two)
  }

  subsets <- rep(FALSE, length(result))
  for (i in 1:length(result)) {
    for (j in 1:length(result)) {
      if (i != j && (!any(result[[i]] & xor(result[[i]], result[[j]])))) {
        subsets[i] <- TRUE
      }
    }
  }

  return(result[!subsets])
}

