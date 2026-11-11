float hash11(float p){ p=fract(p*.1031); p*=p+33.33; p*=p+p; return fract(p); }
float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
vec2 hash22(vec2 p){ vec3 q=fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); q+=dot(q,q.yzx+33.33); return fract((q.xx+q.yz)*q.zy); }
float vnoise(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x),mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x),f.y); }
float fbm(vec2 p,int o){ float a=.5,s=0.; for(int i=0;i<o;i++){ s+=a*vnoise(p); p=mat2(1.6,1.2,-1.2,1.6)*p+7.3; a*=.5;} return s; }
vec2 gauss(vec2 seed){ vec2 h=hash22(seed); float r=sqrt(-2.*log(max(h.x,1e-5))); return r*vec2(cos(6.28318*h.y),sin(6.28318*h.y)); }
vec3 lin2srgb(vec3 c){ c=max(c,0.); return mix(12.92*c,1.055*pow(c,vec3(1./2.4))-.055,step(.0031308,c)); }
vec3 srgb2lin(vec3 c){ return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(.04045,c)); }
float luma(vec3 c){ return dot(c,vec3(.2126,.7152,.0722)); }
mat2 rot(float a){ float c=cos(a),s=sin(a); return mat2(c,s,-s,c); }
