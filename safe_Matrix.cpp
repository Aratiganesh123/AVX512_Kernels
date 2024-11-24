#include <iostream>

template<typename T>

class Matrix{

    public:

    Matrix(int rows, int cols)
    {
        this->rows = rows;
        this->columns = columns;

        p = new T[rows * columns];

        if (p == nullptr) {
            // Error handling for allocation failure
            throw std::bad_alloc();
        }
        
        // Initialize all elements to zero
        for (int i = 0; i < rows * columns; i++) {
            p[i] = T();  // Zero initialization for type T
        }

    }

    int nrows() const {
        return rows;
    }


    int ncols() const {
        return columns;
    }


    ~Matrix() {
        if (p) delete[] p;  // Free the memory
    }


    T* operator[](int r)
    {
        return p + r * columns;
    }


    void print() const {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            std::cout << p[r * columns + c] << " ";
        }
        std::cout << "\n";
    }


    protected:
    int rows;     // Number of rows
    int columns;  // Number of columns
    T* p;         // Pointer to memory
}

};


int main() {
    // Make a matrix of doubles with 3 rows and 4 columns
    Matrix<double> mat(3, 4);

    // Put some values in the matrix
    mat[0][0] = 1.1;
    mat[1][1] = 2.2;
    mat[2][2] = 3.3;

    // Print out the matrix
    for (int r = 0; r < mat.nrows(); r++) {
        for (int c = 0; c < mat.ncols(); c++) {
            printf("%5.2f ", mat[r][c]);
        }
        printf("\n");
    }

    return 0;
}