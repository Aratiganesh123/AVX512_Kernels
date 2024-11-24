#include<iostream>

template<typename T>
class Auto_Ptr
{
    private:
    T* m_ptr{};

    public:
    Auto_Ptr(T* ptr = nullptr) :m_ptr{ptr}
    {

    }
    	~Auto_Ptr()
	{
		delete m_ptr;
	}

    Auto_Ptr& operator=(Auto_Ptr&& a) noexcept
    :m_ptr(a.m_ptr)
    {
        a.m_ptr = nullptr;
    }

    Auto_ptr4& operator=(Auto_ptr4&& a) noexcept
	{
		// Self-assignment detection
		if (&a == this)
			return *this;

		// Release any resource we're holding
		delete m_ptr;

		// Transfer ownership of a.m_ptr to m_ptr
		m_ptr = a.m_ptr;
		a.m_ptr = nullptr; // we'll talk more about this line below

		return *this;
	}


    Auto_Ptr(const Auto_Ptr& a)
    {
        m_ptr = new T;
        *m_ptr = *a.m_ptr;
    }

    Auto_Ptr& operator=(const Auto_Ptr& a)
    {
        if(&a == this)
        {
            return *this;
        }

        delete m_ptr;

        m_ptr = new T;
        *m_ptr = *a.m_ptr;

        return *this;
    }

    T& operator*(){return *m_ptr;}

    T* operator->(){return m_ptr;}

};


class Resource
{
public:
	Resource() { std::cout << "Resource acquired\n"; }
	~Resource() { std::cout << "Resource destroyed\n"; }
};

Auto_Ptr<Resource> generateResource()
{
	Auto_Ptr<Resource> res{new Resource};
	return res; // this return value will invoke the copy constructor
}

int main()
{

	Auto_Ptr<Resource> mainres;
	mainres = generateResource(); // this assignment will invoke the copy assignment

	return 0;
}