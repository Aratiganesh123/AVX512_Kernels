#include<iostream>
#include <algorithm> // for std::max and std::copy_n
using namespace std;


class stringClass
{

    private:
    int m_size {};
    char* m_data{};

    public:
    stringClass(const char* data = nullptr , int size = 0)
    :m_size{std::max(size,0)}
    {
        if(size)
        {
            m_data = new char[static_cast<size_t>(size)];
            std::copy_n(data , size , m_data);
        }
    }

    ~stringClass()
    {
        delete[] m_data;
    }

    stringClass(const stringClass&) = default;

    stringClass& operator=(const stringClass& str);

    friend std::ostream& operator<<(std::ostream& out , const stringClass& obj);
 
};


std::ostream& operator<<(std::ostream& out , const stringClass& obj)
{
    out << obj.m_data;
    return out;
}


stringClass& stringClass::operator=(const stringClass& str)
{
    if(this == &str)
        return *this;

    if(m_data) delete[] m_data;

    m_size = str.m_size;
    m_data = nullptr;

    if(m_size)
    {
        m_data = new char[static_cast<std::size_t>(str.m_size)];
    }

    std::copy_n(str.m_data , m_size , m_data);

    return *this; 
}


int main()
{
	stringClass alex("Alex", 5); // Meet Alex
	stringClass employee("Blex", 5);
    std::cout << employee;
	employee = alex; // Alex is our newest employee
	std::cout << employee; // Say your name, employee

}
	