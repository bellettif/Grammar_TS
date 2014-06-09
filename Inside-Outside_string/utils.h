#ifndef UTILS_H
#define UTILS_H

#define SEQ_DEBUG

#ifdef SEQ_DEBUG

#define SEQ_COUT(msg) std::cout << msg << std::endl
#define SEQ_CERR(msg) std::cerr << msg << std::endl
#define SEQ_ASSERT(cond) assert(cond)

#else

#define SEQ_COUT(msg)
#define SEQ_CERR(msg)
#define SEQ_ASSERT(cond)

#endif


#endif // UTILS_H
