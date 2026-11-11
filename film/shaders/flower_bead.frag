#version 410
// a sessile drop, ray traced. it is a wide-angle lens with the sheet for a floor:
// straight through it shows the petal, big; toward its edge it shows the sky, upside down.
#include "flower_common.glsl"
in vec3 vC,vNb,vTb,vBb; in vec4 vS; in vec4 vU; in vec2 vW;
out vec4 o;
uniform sampler2D uLit,uPig,uEnv;
uniform vec3 uSunD,uSunE,uGround;
uniform vec2 uRes;
uniform float uKcoc,uPxPerUnit,uN,uW;
const float PI=3.14159265;
const float ETA=1./1.333;

vec3 envAll(vec3 r,float lod,float sunW){
  float az=atan(r.x,r.z);
  vec3 c;
  if(r.y>=0.) c=textureLod(uEnv,vec2(az/(2.*PI)+.5,asin(min(r.y,1.))/(.5*PI)),lod).rgb;
  else {
    vec3 h=textureLod(uEnv,vec2(az/(2.*PI)+.5,.004),lod).rgb;
    c=mix(uGround,h*.45,exp(r.y/.012)*.8);
  }
  // the sun, in a drop: the sharpest thing in the frame
  float th=length(r+uSunD);
  c+=uSunE*exp(-th*th/(2.*sunW*sunW))/(2.*PI*sunW*sunW);
  // the brightest of the cloud is behind whoever is holding the camera, up and to the left
  float tc=length(r-vec3(-.46,.62,-.63));
  c+=vec3(.92,.96,1.)*9.*exp(-tc*tc/(2.*.17*.17));
  return c;
}
float fres(float c){
  float g=sqrt(max(1.7689-1.+c*c,0.));
  float a=(g-c)/(g+c);
  float b=(c*(g+c)-1.)/(c*(g-c)+1.);
  return .5*a*a*(1.+b*b);
}
void main(){
  // ray for this pixel through this piece of the lens
  vec2 ndc=(gl_FragCoord.xy/uRes*2.-1.)-uJit;
  vec2 view=ndc*vec2(uAspect,1.)/uZoom+uGc;
  float s=1./uFd;
  vec3 Pf=uC+uF*s+(uR*view.x+uU*view.y)*uTan*s;
  vec3 O=uC+uR*uLens.x+uU*uLens.y;
  vec3 d=normalize(Pf-O);
  // into the drop's frame, where it is a unit sphere with its centre at height zc, cut by z=0
  float R=vS.x,e=vS.y,zc=vS.w;
  float k=R<.0012?.97:mix(.97,.82,clamp((R-.0012)/.0012,0.,1.));   // gravity flattens the big ones
  vec3 sc=vec3(R,R*e,R*k);
  vec3 tb=normalize(vTb),bb=normalize(vBb),nb=normalize(vNb);
  // start the ray beside the drop: the smallest of these are a tenth of a millimetre and thirty centimetres away,
  // and a float can't take the difference of those squares
  vec3 Os=O+d*dot(vC-O,d);
  vec3 ro=vec3(dot(Os-vC,tb),dot(Os-vC,bb),dot(Os-vC,nb))/sc;
  vec3 rd=vec3(dot(d,tb),dot(d,bb),dot(d,nb))/sc;
  float dl=length(rd); rd/=dl;
  vec3 oc=ro-vec3(0.,0.,zc);
  float b=dot(oc,rd),cc=dot(oc,oc)-1.,disc=b*b-cc;
  if(disc<=0.) discard;
  float tn=-b-sqrt(disc);
  vec3 q=ro+rd*tn;
  // the contact line isn't a circle
  float ang=atan(q.y,q.x);
  float cl=.05*sin(ang*2.+vU.x*40.)+.035*sin(ang*3.+vU.y*60.)+.02*sin(ang*5.+vU.x*90.);
  if(q.z<cl) discard;
  vec3 hit=vC+sc.x*q.x*tb+sc.y*q.y*bb+sc.z*q.z*nb;
  float hz=dot(hit-uC,uF);
  gl_FragDepth=depthOf(hz);
  vec3 nl=q-vec3(0.,0.,zc);
  vec3 nw=normalize(nl.x/sc.x*tb+nl.y/sc.y*bb+nl.z/sc.z*nb);
  float c0=max(dot(-d,nw),0.);
  float F=fres(c0);
  // how much the sun's image must be spread so that a blurred drop is a disc, not a swarm
  float coc=uKcoc*abs(uFd-1./hz)*uZoom*uPxPerUnit;
  float rpx=R/(hz*uTan)*uZoom*uPxPerUnit;
  float sunW=.012+1.7*coc/max(rpx,1.)/sqrt(uN);
  float lod=clamp(log2(max(coc*.9/sqrt(uN),1.))+2.,2.,7.);
  // whatever points down from here lands on the sheet, not in the sky: that is what stops a drop being a soap bubble
  vec3 rv=reflect(d,nw);
  vec3 refl=mix(texture(uLit,vU.xy+vec2(q.x*sc.x*vU.z,q.y*sc.y*vU.w),lod).rgb*.85,envAll(rv,lod+1.,sunW),smoothstep(-.03,.10,dot(rv,nb)));
  // inside
  vec3 dt=refract(d,nw,ETA);
  vec3 ldt=vec3(dot(dt,tb),dot(dt,bb),dot(dt,nb))/sc;
  float ll=length(ldt); ldt/=ll;
  vec3 q0=q+ldt*1e-4;
  vec3 c=vec3(0.);
  vec3 thru=vec3(1.);
  float ior=1.333;
  for(int bounce=0;bounce<3;bounce++){
    vec3 oc2=q0-vec3(0.,0.,zc);
    float b2=dot(oc2,ldt),c2=dot(oc2,oc2)-1.,dd=b2*b2-c2;
    float ts=-b2+sqrt(max(dd,0.));
    float tp=ldt.z<-1e-5?-q0.z/ldt.z:1e9;
    if(tp<ts){
      // the floor: the petal, lit from behind, seen through a lens
      vec3 h=q0+ldt*tp;
      vec2 off=vec2(h.x*sc.x*vU.z,h.y*sc.y*vU.w);
      vec2 uv=vU.xy+off;
      vec3 p=texture(uLit,uv,lod-2.).rgb;
      float al=textureLod(uPig,uv,2.).a;
      if(al<.5) p=envAll(normalize(d),5.,.08)*.8;
      // the sheet is lit from behind, and what comes through it is mostly still going the way it was:
      // down the middle of the drop you look along that; out toward its edge you look across it
      float ra=length(h.xy)/sqrt(max(1.-zc*zc,.05));
      float ob=abs(ldt.z);
      float limb=mix(1.34,.60,smoothstep(.30,.92,ra))*mix(.82,1.,smoothstep(.62,.95,ob));
      c+=thru*p*limb*vec3(1.,mix(.90,1.,limb/1.34),mix(.70,1.,limb/1.34));
      break;
    }
    vec3 h=q0+ldt*ts;
    vec3 nin=h-vec3(0.,0.,zc);
    vec3 wn=normalize(nin.x/sc.x*tb+nin.y/sc.y*bb+nin.z/sc.z*nb);
    vec3 wd=normalize(ldt.x*sc.x*tb+ldt.y*sc.y*bb+ldt.z*sc.z*nb);
    float ci=max(dot(wd,wn),0.);
    float sin2=(1.-ci*ci)*1.777;
    float Fi=sin2>=1.?1.:fres(sqrt(max(1.-sin2,0.)));
    if(sin2<1.){
      vec3 od=refract(wd,-wn,1.333);
      float oz=dot(od,nb);
      vec3 ex=envAll(od,lod,sunW);
      if(oz<.06){
        vec3 pl=texture(uLit,vU.xy+vec2(h.x*sc.x*vU.z,h.y*sc.y*vU.w),lod).rgb*.55;
        ex=mix(pl,ex,smoothstep(-.04,.06,oz));
      }
      c+=thru*(1.-Fi)*ex;
    }
    if(Fi<.04) break;
    thru*=Fi;
    vec3 rd2=reflect(wd,-wn);
    ldt=normalize(vec3(dot(rd2,tb),dot(rd2,bb),dot(rd2,nb))/sc);
    q0=h+ldt*1e-4;
  }
  o=vec4(c*(1.-F)+refl*F,1.)*uW;
}
