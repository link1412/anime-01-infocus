#version 410
// A path tracer that writes to surfaces, not to a camera.
// Every texel of the atlas gathers the light arriving at its surface point (irradiance / pi), kept in separate
// passes so that brightness and colour can be chosen per frame: the screen as four bilinear corners, and the window.
layout(location=0) out vec4 oA;
layout(location=1) out vec4 oB;
layout(location=2) out vec4 oC;
layout(location=3) out vec4 oD;
layout(location=4) out vec4 oW;
uniform sampler2D uGP,uGN;
uniform float uBatch;
uniform int uSpp;
#include "desk_geo.glsl"

uint seed;
float rnd(){
  seed=seed*747796405u+2891336453u;
  uint w=((seed>>((seed>>28u)+4u))^seed)*277803737u;
  return float((w>>22u)^w)*(1./4294967296.);
}
vec3 cosDir(vec3 n,float a,float b){
  float z=1.-2.*a; float r=sqrt(max(1.-z*z,0.)); float f=6.2831853*b;
  return normalize(n+vec3(r*cos(f),r*sin(f),z));
}
bool visible(vec3 a,vec3 b){
  vec3 d=b-a; float L=length(d); d/=L;
  float t=0.;
  float lim=L-.001;
  for(int i=0;i<64;i++){
    float h=map(a+d*t);
    if(h<.0002) return false;
    t+=h;
    if(t>=lim) return true;
  }
  return true;
}

void main(){
  ivec2 ip=ivec2(gl_FragCoord.xy);
  vec4 gp=texelFetch(uGP,ip,0);
  oA=oB=oC=oD=oW=vec4(0.,0.,0.,1.);
  if(gp.w<.5) return;
  vec3 n0=normalize(texelFetch(uGN,ip,0).xyz);
  vec3 p0=gp.xyz+n0*.0012;
  if(map(p0)<.0002) return;          // this surface is pressed against something
  seed=uint(ip.x)*73856093u^uint(ip.y)*19349663u^uint(uBatch)*83492791u;
  // the same low-discrepancy points for all paths, turned differently per texel
  vec2 cp=vec2(rnd(),rnd()),cq=vec2(rnd(),rnd());
  vec3 sum[5];
  for(int i=0;i<5;i++) sum[i]=vec3(0.);
  float area=4.*HW*HH;
  vec2 wc=.5*(WINDOW.xy+WINDOW.zw),wh=.5*(WINDOW.zw-WINDOW.xy);
  float wArea=4.*wh.x*wh.y;
  for(int s=0;s<uSpp;s++){
    float idx=uBatch*float(uSpp)+float(s);
    vec2 r2=fract(cp+idx*vec2(.7548776662,.5698402910));
    vec2 r3=fract(cq+idx*vec2(.5698402910,.7548776662));
    vec3 p=p0,n=n0,thr=vec3(1.);
    for(int b=0;b<6;b++){
      // the screen, a Lambertian rectangle facing +z
      vec2 u=b==0?r2:vec2(rnd(),rnd());
      vec3 sp=vec3((u-.5)*2.*vec2(HW,HH),.001);
      vec3 w=sp-p; float d2=dot(w,w); vec3 wd=w/sqrt(d2);
      float cP=dot(n,wd),cS=-wd.z;
      if(p.z>.0003&&cP>0.&&cS>0.&&visible(p,sp)){
        // close to the glass 1/d^2 explodes: the true mean is bounded by one, so bound the estimate
        float g=min(cP*cS*area/(d2*3.14159265),b==0?6.:1.);
        vec3 e=thr*g;
        sum[0]+=e*(1.-u.x)*(1.-u.y);
        sum[1]+=e*u.x*(1.-u.y);
        sum[2]+=e*(1.-u.x)*u.y;
        sum[3]+=e*u.x*u.y;
      }
      // the window on the wall behind the operator
      vec2 v=b==0?r3:vec2(rnd(),rnd());
      vec3 wp=vec3(wc+(v-.5)*2.*wh,RMAX.z-.001);
      w=wp-p; d2=dot(w,w); wd=w/sqrt(d2);
      cP=dot(n,wd);
      float cW=wd.z;
      if(cP>0.&&cW>0.&&visible(p,wp)) sum[4]+=thr*cP*cW*wArea/(d2*3.14159265);
      // one more bounce
      vec3 dir=cosDir(n,rnd(),rnd());
      float t; int m; vec3 hn;
      if(!traceRoom(p,dir,1e3,72,t,m,hn)) break;
      vec3 q=p+dir*t;
      thr*=bounceAlbedo(m);
      n=hn;
      p=q+n*.0012;
      float pk=max(thr.x,max(thr.y,thr.z));
      if(b>1){
        if(rnd()>pk) break;
        thr/=pk;
      }
    }
  }
  float k=1./float(uSpp);
  oA.rgb=sum[0]*k; oB.rgb=sum[1]*k; oC.rgb=sum[2]*k; oD.rgb=sum[3]*k; oW.rgb=sum[4]*k;
}
