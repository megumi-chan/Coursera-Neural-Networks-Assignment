function ret = q11(rbm_w)
    len = size(rbm_w, 1);
    A = dec2bin(0:2^len - 1)-'0';
    ret = log(sum(prod(1 + exp(rbm_w'*A'))));
end