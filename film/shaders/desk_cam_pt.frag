#version 410
// The one solid thing in the glass. Traced once, slowly, the way a product photograph is lit:
// a camera built from distance fields, a dark room around it, a dim monitor in front and a window behind.
// Rays start at the eye and leave through the glass plane, so this is what the mirror really shows.
in vec2 uv; layout(location=0) out vec4 o;
uniform vec3 uO,uF,uR,uU;
uniform float uTan,uAspect,uScale,uBatch;
uniform vec2 uRes;
uniform sampler2D uEngrave;

const float PI=3.14159265;
const float BODY=1.,GRAIN=2.,BARREL=3.,RIBBED=4.,METAL=5.,SATIN=6.,GLASS=7.,PHONE=8.,LAMP=9.,DARK=10.,FRONT=11.;
const vec2 LC=vec2(-.006,-.003);       // the lens axis, in the camera's own coordinates
const float ZG=.0362,RG=.0362;         // the front element: a sphere, centre and radius

uint seed;
float rnd(){
  seed=seed*747796405u+2891336453u;
  uint w=((seed>>((seed>>28u)+4u))^seed)*277803737u;
  return float((w>>22u)^w)*(1./4294967296.);
}
float hash21(vec2 p){ vec3 q=fract(vec3(p.xyx)*.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
vec2 hash22(vec2 p){ vec3 q=fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); q+=dot(q,q.yzx+33.33); return fract((q.xx+q.yz)*q.zy); }
float vnoise(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x),mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x),f.y); }

float sdRB(vec3 p,vec3 b,float r){ vec3 q=abs(p)-b+r; return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.)-r; }
float sdRect(vec2 p,vec2 b,float r){ vec2 q=abs(p)-b+r; return length(max(q,0.))+min(max(q.x,q.y),0.)-r; }
// a shape d2 in its plane, h either side of it along its axis, with its edges broken by f
float slab(float d2,float z,float h,float f){ vec2 w=vec2(d2,abs(z)-h)+f; return min(max(w.x,w.y),0.)+length(max(w,0.))-f; }
float ring(vec2 rz,float r1,float z0,float z1,float f){ return slab(rz.x-r1,rz.y-.5*(z0+z1),.5*(z1-z0),f); }
float smin(float a,float b,float k){ float h=clamp(.5+.5*(b-a)/k,0.,1.); return mix(b,a,h)-k*h*(1.-h); }
vec2 U(vec2 a,vec2 b){ return a.x<b.x?a:b; }

// nobody holds a camera dead square to anything
vec3 toCam(vec3 p){
  const float ya=.052,pi=-.040;
  p.xz=mat2(cos(ya),-sin(ya),sin(ya),cos(ya))*p.xz;
  p.yz=mat2(cos(pi),-sin(pi),sin(pi),cos(pi))*p.yz;
  return p;
}
vec3 fromCam(vec3 p){
  const float ya=.052,pi=-.040;
  p.yz=mat2(cos(pi),sin(pi),-sin(pi),cos(pi))*p.yz;
  p.xz=mat2(cos(ya),sin(ya),-sin(ya),cos(ya))*p.xz;
  return p;
}
vec3 toPhone(vec3 p){
  p-=vec3(-.058,.030,-.056);
  const float ya=.060,ro=.024;
  p.xz=mat2(cos(ya),-sin(ya),sin(ya),cos(ya))*p.xz;
  p.xy=mat2(cos(ro),-sin(ro),sin(ro),cos(ro))*p.xy;
  return p;
}

vec2 map(vec3 pw){
  vec3 p=toCam(pw);
  // the casting and the grip growing out of it
  float bx=sdRB(p,vec3(.0620,.0355,.0205),.0058);
  float gr=sdRB(p-vec3(.0495,-.0040,.0258),vec3(.0128,.0300,.0138),.0102);
  float body=smin(bx,gr,.0058);
  // the seam where the top plate meets the skin
  body=max(body,-max(abs(p.y-.0176)-.00030,-(body+.0008)));
  vec2 r=vec2(body,BODY);
  // top plate: a dial, the shutter with its collar, the shoe
  vec3 q=p-vec3(.0268,.0376,-.0040);
  float th=atan(q.z,q.x);
  r=U(r,vec2(slab(length(q.xz)-.0112+.00030*smoothstep(-.4,.4,sin(th*58.)),q.y,.0046,.0006),SATIN));
  q=p-vec3(.0486,.0366,.0088);
  r=U(r,vec2(slab(length(q.xz)-.0084,q.y,.0030,.0007),SATIN));
  r=U(r,vec2(slab(length(q.xz)-.0050,q.y-.0028,.0023,.0011),METAL));
  q=p-vec3(-.0060,.0368,-.0020);
  r=U(r,vec2(sdRB(q,vec3(.0118,.0024,.0118),.0011),DARK));
  q.x=abs(q.x)-.0096;
  r=U(r,vec2(sdRB(q-vec3(0.,.0030,0.),vec3(.0017,.0011,.0108),.0004),METAL));
  // front of the casting: the assist lamp, the release, strap lugs
  q=p-vec3(.0338,.0226,.0208);
  r=U(r,vec2(slab(length(q.xy)-.0031,q.z,.0008,.0003),LAMP));
  q=p-vec3(.0300,-.0176,.0208);
  r=U(r,vec2(slab(length(q.xy)-.0050,q.z-.0005,.0013,.0006),DARK));
  q=p; q.x=abs(q.x)-.0648; q-=vec3(0.,.0220,.0);
  r=U(r,vec2(slab(abs(length(q.yz)-.0031)-.0009,q.x,.0011,.0004),SATIN));
  // the lens, turned on a lathe
  q=p-vec3(LC,0.);
  float rad=length(q.xy);
  th=atan(q.y,q.x);
  vec2 rz=vec2(rad,q.z);
  r=U(r,vec2(ring(rz,.0304,.0190,.0240,.0005),METAL));
  float lens=ring(rz,.0318,.0235,.0325,.0006);
  r=U(r,vec2(lens,BARREL));
  float rib=.00036*smoothstep(-.45,.45,sin(th*128.));
  r=U(r,vec2(ring(vec2(rad+rib,q.z),.0338,.0325,.0582,.0010),RIBBED));
  // the rim, and the bowl cut into it
  float Ri=mix(.0198,.0278,clamp((q.z-.0655)/(.0764-.0655),0.,1.));
  float cut=max((rad-Ri)*.80,.0655-q.z);
  r=U(r,vec2(max(ring(vec2(rad+(q.z-.0582)*.13,q.z),.0324,.0582,.0724,.0007),-cut),BARREL));
  float rim=ring(rz,.0306,.0720,.0764,.0013);
  r=U(r,vec2(max(rim,-cut),FRONT));
  r=U(r,vec2(slab(abs(rad-.0205)-.0007,q.z-.0662,.0011,.0003),DARK));
  r=U(r,vec2(max(length(q-vec3(0.,0.,ZG))-RG,rad-.0200),GLASS));
  // and behind it all, the phone, held to the eyepiece a little crooked
  vec3 h=toPhone(pw);
  float d2=sdRect(h.xy,vec2(.0775,.0375),.0100);
  r=U(r,vec2(slab(d2,h.z,.0042,.0015),PHONE));
  return r;
}
vec3 normal(vec3 p){
  vec2 e=vec2(.00012,0.);
  return normalize(vec3(map(p+e.xyy).x-map(p-e.xyy).x,map(p+e.yxy).x-map(p-e.yxy).x,map(p+e.yyx).x-map(p-e.yyx).x));
}
bool march(vec3 ro,vec3 rd,float t0,float t1,out float t){
  t=t0;
  for(int i=0;i<140;i++){
    float d=map(ro+rd*t).x;
    if(d<.00007){ return true; }
    t+=d*.92;
    if(t>t1) return false;
  }
  return false;
}
// entry and exit of a box around everything
vec2 bounds(vec3 ro,vec3 rd){
  const vec3 lo=vec3(-.158,-.062,-.085),hi=vec3(.086,.085,.088);
  vec3 a=(lo-ro)/rd,b=(hi-ro)/rd;
  vec3 m=min(a,b),x=max(a,b);
  return vec2(max(max(m.x,m.y),max(m.z,0.)),min(x.x,min(x.y,x.z)));
}

// ---------------------------------------------------------------- the room, as seen from the camera.
// lengths are the camera's own (the glass flatters it by uScale). the monitor is ahead, the window far behind.
const float MD=.60;
const vec3 CMON=vec3(.60,.72,1.);
vec3 roomS(vec3 o_,vec3 d,float soft){
  vec3 c;
  float up=d.y;
  if(d.z>0.){
    // what's ahead: the wall behind the monitor, the last of the window's light on it; the desk below it
    c=mix(vec3(.022,.021,.020),vec3(.040,.034,.028),smoothstep(-.1,.7,up));
    c=mix(c,vec3(.016,.011,.008),smoothstep(-.10,-.40,up));
    float t=(MD-o_.z)/d.z;
    vec2 h=o_.xy+d.xy*t;
    float m=smoothstep(soft,-soft,sdRect(h,vec2(.186,.105),.003));
    float bez=smoothstep(soft,-soft,sdRect(h,vec2(.194,.113),.006));
    c=mix(c,vec3(.003),bez);
    c=mix(c,CMON*.15,m);
  }else{
    // behind: the back wall, the shape of the operator, and a window with the town's light on low cloud in it
    c=mix(vec3(.014,.016,.020),vec3(.030,.032,.040),smoothstep(-.4,.8,up));
    float t=(-1.25-o_.z)/d.z;
    vec2 h=o_.xy+d.xy*t;
    float w=smoothstep(.02,-.02,sdRect(h-vec2(-.13,.40),vec2(.36,.46),.01));
    vec3 sky=mix(vec3(.62,.40,.25),vec3(.20,.26,.38),clamp((h.y+.05)/.9,0.,1.));
    c=mix(c,sky,w);
    // whoever is holding this, close enough to fill the middle of the view behind it
    float t2=(-.20-o_.z)/d.z;
    vec2 h2=o_.xy+d.xy*t2;
    float them=min(length((h2-vec2(0.,.11))/vec2(.095,.125))-1.,sdRect(h2-vec2(0.,-.20),vec2(.20,.20),.08)/.12);
    c=mix(vec3(.006),c,smoothstep(-.06,.06,them));
  }
  return c;
}

vec3 room(vec3 o_,vec3 d){ return roomS(o_,d,.006); }

// ---------------------------------------------------------------- the front element
// looking straight into a lens you look at a stack of curved mirrors, each coated a different colour,
// every one holding its own tiny picture of whatever is bright in front of it
vec3 lensStack(vec3 P,vec3 d,vec3 Pw){
  vec3 n=normalize(P-vec3(0.,0.,ZG));
  float c=max(-dot(d,n),0.);
  const float G=15.;   // what the coatings let through of a dim screen, to something looking for it
  vec3 col=vec3(0.);
  vec3 r0=reflect(d,n);
  float F=.0026+.11*pow(1.-c,5.);
  vec3 coat=mix(vec3(.30,1.,.52),vec3(1.,.42,.86),smoothstep(.015,.17,1.-c));
  col+=room(Pw,fromCam(r0))*coat*F*G;
  vec3 di=refract(d,n,1./1.62);
  // internal surfaces: centre on the axis, signed radius, tint
  const int NS=6;
  float zc[6]=float[6](-.052,.010,.052,-.034,.078,-.080);
  float rr[6]=float[6](.082,-.046,-.016,.058,-.030,.120);
  // none of them is focused where the front surface is: each picture is soft by its own amount
  float ss[6]=float[6](.05,.10,.016,.07,.03,.16);
  vec3 tt[6]=vec3[6](vec3(1.,.42,.86),vec3(.40,1.,.62),vec3(1.,.62,.26),vec3(.38,.62,1.),vec3(.62,1.,.40),vec3(.90,.40,.56));
  float gg[6]=float[6](.0016,.0030,.0052,.0012,.0024,.0005);
  for(int i=0;i<NS;i++){
    vec3 oc=P-vec3(0.,0.,zc[i]);
    float R=abs(rr[i]);
    float b=dot(oc,di),h=b*b-dot(oc,oc)+R*R;
    if(h<0.) continue;
    float t=-b+(rr[i]>0.?sqrt(h):-sqrt(h));
    if(t<=.001) t=-b+sqrt(h);
    if(t<=.001) continue;
    vec3 Q=P+di*t;
    vec3 nI=normalize(Q-vec3(0.,0.,zc[i]));
    vec3 rI=reflect(di,nI);
    rI=normalize(vec3(rI.xy*1.55,rI.z));
    if(rI.z<=.05) continue;
    // the deeper a ghost, the more of it the barrel has clipped
    float clip=smoothstep(.0190,.0130,length(Q.xy));
    col+=roomS(Pw,fromCam(rI),ss[i])*tt[i]*gg[i]*G*clip;
  }
  // down the barrel: baffles, then the iris, wide open, then nothing
  float tz=(.0350-P.z)/min(di.z,-.05);
  vec3 I=P+di*tz;
  float ri=length(I.xy);
  float a=atan(I.y,I.x);
  float k=2.*PI/9.;
  float blade=cos(mod(a+.2,k)-.5*k)*ri;
  float open=smoothstep(.0150,.0156,blade);
  float baffle=.5+.5*sin(ri*2600.);
  col+=vec3(.0010,.0011,.0013)*(.4+.6*baffle)*open;
  col+=vec3(.0016,.0010,.0024)*exp(-ri*ri/(2.*.0060*.0060))*(1.-open);
  return col;
}

// ---------------------------------------------------------------- surfaces
float pebble(vec2 p){
  // the skin: rounded pebbles and the valleys between them
  p*=1./.00125;
  vec2 g=floor(p),f=fract(p);
  float a=1.,b=1.;
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
    vec2 c=vec2(x,y);
    vec2 h=hash22(g+c)*.85+.075;
    float d=length(c+h-f);
    if(d<a){ b=a; a=d; }else if(d<b){ b=d; }
  }
  return smoothstep(.0,.30,b-a)*(.75+.5*vnoise(p*.22));
}
vec3 bumped(vec3 n,vec3 p,float amp){
  // pick the plane the surface mostly lies in
  vec3 an=abs(n);
  vec2 u;
  float e=.00018;
  float dx,dy;
  if(an.z>an.x&&an.z>an.y){ u=p.xy; dx=pebble(u+vec2(e,0.))-pebble(u-vec2(e,0.)); dy=pebble(u+vec2(0.,e))-pebble(u-vec2(0.,e)); return normalize(n-amp*vec3(dx,dy,0.)); }
  if(an.x>an.y){ u=p.zy; dx=pebble(u+vec2(e,0.))-pebble(u-vec2(e,0.)); dy=pebble(u+vec2(0.,e))-pebble(u-vec2(0.,e)); return normalize(n-amp*vec3(0.,dy,dx)); }
  u=p.xz; dx=pebble(u+vec2(e,0.))-pebble(u-vec2(e,0.)); dy=pebble(u+vec2(0.,e))-pebble(u-vec2(0.,e)); return normalize(n-amp*vec3(dx,0.,dy));
}
vec3 cosDir(vec3 n){
  float a=rnd(),b=rnd();
  float z=1.-2.*a; float r=sqrt(max(1.-z*z,0.)); float f=2.*PI*b;
  return normalize(n+vec3(r*cos(f),r*sin(f),z)*.999);
}
// visible normals of GGX (Heitz 2018)
vec3 ggxH(vec3 n,vec3 v,float al){
  vec3 up=abs(n.z)<.99?vec3(0.,0.,1.):vec3(1.,0.,0.);
  vec3 t=normalize(cross(up,n)),b=cross(n,t);
  vec3 Ve=vec3(dot(v,t),dot(v,b),dot(v,n));
  vec3 Vh=normalize(vec3(al*Ve.x,al*Ve.y,max(Ve.z,0.)));
  float l2=dot(Vh.xy,Vh.xy);
  vec3 T1=l2>1e-7?vec3(-Vh.y,Vh.x,0.)/sqrt(l2):vec3(1.,0.,0.);
  vec3 T2=cross(Vh,T1);
  float r=sqrt(rnd()),ph=2.*PI*rnd();
  float x=r*cos(ph),y=r*sin(ph);
  float s=.5*(1.+Vh.z);
  y=(1.-s)*sqrt(max(1.-x*x,0.))+s*y;
  vec3 Nh=x*T1+y*T2+sqrt(max(1.-x*x-y*y,0.))*Vh;
  vec3 Ne=normalize(vec3(al*Nh.x,al*Nh.y,max(Nh.z,0.)));
  return normalize(t*Ne.x+b*Ne.y+n*Ne.z);
}
float G1(float c,float al){ float a2=al*al; return 2.*c/(c+sqrt(a2+(1.-a2)*c*c)); }

struct Surf{ vec3 n,alb,F0; float rough; };
Surf surface(float mat,vec3 Pw,vec3 n){
  vec3 p=toCam(Pw);
  Surf s; s.n=n; s.alb=vec3(.022); s.F0=vec3(.045); s.rough=.42;
  vec3 q=p-vec3(LC,0.);
  float rad=length(q.xy);
  if(mat==BODY){
    // above the seam it's painted magnesium. below it, on the front and round the grip, it's skin
    bool skin=p.y<.0176&&(n.z>.05||p.x>.030);
    if(skin){
      s.alb=vec3(.0165,.0160,.0155); s.rough=.50; s.F0=vec3(.042);
      s.n=bumped(n,p,.28);
    }else{
      float g=vnoise(p.xy*5200.+p.z*4100.)-.5;
      s.alb=vec3(.026,.026,.027); s.rough=.33+.05*g; s.F0=vec3(.050);
      s.n=normalize(n+.018*vec3(g,vnoise(p.yz*5300.)-.5,0.));
    }
  }else if(mat==BARREL){
    s.alb=vec3(.020); s.rough=.36; s.F0=vec3(.048);
    float g=vnoise(vec2(atan(q.y,q.x)*90.,q.z*2300.))-.5;
    s.n=normalize(n+.012*g*normalize(vec3(q.xy,0.)));
  }else if(mat==RIBBED){
    s.alb=vec3(.011); s.rough=.55; s.F0=vec3(.04);
  }else if(mat==METAL){
    s.alb=vec3(0.); s.F0=vec3(.62,.60,.58); s.rough=.24;
  }else if(mat==SATIN){
    s.alb=vec3(0.); s.F0=vec3(.20,.20,.205); s.rough=.34;
  }else if(mat==LAMP){
    s.alb=vec3(.006,.002,.002); s.F0=vec3(.030); s.rough=.10;
  }else if(mat==DARK){
    s.alb=vec3(.008); s.F0=vec3(.045); s.rough=.38;
  }else if(mat==FRONT){
    // the bowl: fine turned grooves, a thread by the rim, and the lens's name in white
    s.alb=vec3(.0085); s.F0=vec3(.045); s.rough=.30;
    float lines=sin(rad*2.*PI/.00040);
    if(rad>.0212&&rad<.0282){
      vec3 rd_=normalize(vec3(q.xy,0.));
      float amp=rad>.0258?.34:.10;
      s.n=normalize(n+amp*lines*rd_);
    }
    if(rad>.0226&&rad<.0256){
      float th=atan(q.y,q.x);
      float e=texture(uEngrave,vec2(fract(th/(2.*PI)+.31),(rad-.0226)/.0030)).r;
      s.alb=mix(s.alb,vec3(.70),e);
      s.rough=mix(s.rough,.6,e);
    }
  }else if(mat==PHONE){
    vec3 h=toPhone(Pw);
    float d2=sdRect(h.xy,vec2(.0775,.0375),.0100);
    if(d2>-.0012){ s.alb=vec3(0.); s.F0=vec3(.34,.34,.36); s.rough=.27; }
    else{
      // it has been held: prints and the haze of a pocket
      float haze=vnoise(h.xy*150.+3.)*.6+vnoise(h.xy*420.)*.3+vnoise(h.xy*1300.)*.1;
      float print_=smoothstep(.46,.86,haze)*(.7+.3*sin(length(h.xy-vec2(-.03,.012))*16000.));
      s.alb=vec3(.0065,.0068,.0078); s.F0=vec3(.034); s.rough=.08+.055*print_;
    }
  }
  return s;
}

vec3 path(vec3 ro,vec3 rd,float t0,float t1,out float hit){
  vec3 L=vec3(0.),thr=vec3(1.);
  hit=0.;
  float t;
  for(int b=0;b<4;b++){
    vec2 bd=b==0?vec2(t0,t1):bounds(ro,rd);
    if(bd.y<bd.x||!march(ro,rd,max(bd.x,0.),bd.y,t)){
      if(b>0) L+=thr*room(ro,rd);
      break;
    }
    if(b==0) hit=1.;
    vec3 p=ro+rd*t;
    float mat=map(p).y;
    vec3 ng=normal(p);
    if(mat==GLASS){
      vec3 pc=toCam(p)-vec3(LC,0.);
      vec3 dc=toCam(rd);
      L+=thr*lensStack(pc,dc,p);
      break;
    }
    Surf s=surface(mat,p,ng);
    vec3 v=-rd;
    float al=max(s.rough*s.rough,.002);
    vec3 h=ggxH(s.n,v,al);
    float vh=max(dot(v,h),0.);
    vec3 F=s.F0+(1.-s.F0)*pow(1.-vh,5.);
    float Fa=dot(F,vec3(.333));
    float ps=clamp(Fa*(1.+.5*step(dot(s.alb,vec3(.333)),.001))+.25,.25,.92);
    if(dot(s.alb,vec3(1.))<=0.) ps=1.;
    vec3 nd;
    if(rnd()<ps){
      nd=reflect(rd,h);
      float nl=dot(s.n,nd),nv=max(dot(s.n,v),1e-3);
      if(nl<=0.) break;
      thr*=F*G1(nl,al)/ps;
    }else{
      nd=cosDir(s.n);
      thr*=s.alb*(1.-F)/(1.-ps);
    }
    if(dot(nd,ng)<=0.) break;
    ro=p+ng*.00022;
    rd=nd;
  }
  return L;
}

void main(){
  ivec2 ip=ivec2(gl_FragCoord.xy);
  seed=uint(ip.x)*73856093u^uint(ip.y)*19349663u^(uint(uBatch)*83492791u+1u);
  rnd();rnd();
  vec2 j=vec2(rnd(),rnd())+vec2(rnd(),rnd())-1.;   // tent
  vec2 st=(gl_FragCoord.xy+.5*j)/uRes;
  vec2 v=(st*2.-1.)*vec2(uAspect,1.);
  vec3 rd=normalize(uF+v.x*uTan*uR+v.y*uTan*uU);
  // following the ray through the mirror: the eye sits twice the distance away, and the glass flatters by uScale
  vec3 ro=vec3(0.,0.,2.*uO.z)/uScale;
  vec2 bd=bounds(ro,rd);
  if(bd.y<bd.x){ o=vec4(0.); return; }
  float hit;
  vec3 c=path(ro,rd,bd.x,bd.y,hit);
  o=vec4(min(c,vec3(1.5))*hit,hit);
}
