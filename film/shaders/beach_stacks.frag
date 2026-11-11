#version 410
// Basalt out at sea, raymarched once: flat-topped columns stand side by side and fall away in steps,
// which is all columnar jointing is. Seen against the light, through as much air as lies on the way.
in vec2 uv; out vec4 o;
#include "common.glsl"
const float PI=3.14159265;
const int NS=5;
uniform vec4 uRange;
uniform float uH,uCurv,uPix;
uniform vec3 uSun,uSunE;
uniform float uSkyLevel;
uniform sampler2D uBg;
uniform vec4 uS[NS];    // x,z,radius,height
uniform vec4 uS2[NS];   // elongation,angle,seed,column width

float sat(float x){ return clamp(x,0.,1.); }

float body(vec2 p,int i){
  vec4 a=uS[i],b=uS2[i];
  vec2 q=rot(b.y)*(p-a.xy)/(a.z*vec2(b.x,1.));
  // cliffs all round but never plumb: buttresses, notches, and a shoulder lower on one side
  float l=length(q)+.6*(fbm(p/(a.z*.75)+b.z,3)-.44)+.2*(fbm(p/(a.z*.17)+b.z+3.,3)-.42);
  float taper=pow(sat(1.-l),.62);
  float crown=fbm(p/(a.z*.6)+b.z+7.,4);
  float h=a.w*taper*(.36+1.45*max(crown-.2,0.))*1.25;
  float skirt=a.w*.05*smoothstep(1.14,.95,l)*(.2+1.6*fbm(p/(a.z*.25)+b.z+11.,2));
  return max(h,skirt);
}
float headland(vec2 p,float az,float r){
  if(az<.3) return -1.;
  float rc=2450.-620.*sin(sat((az-.3)*.95)*1.57)+260.*(fbm(vec2(az*5.,1.3),3)-.45);
  float x=(r-rc)/330.;
  float nose=az+.05*(fbm(vec2(r/260.,4.),3)-.4);
  float mask=smoothstep(.4,.455,nose)*(1.-smoothstep(1.32,1.46,az));
  // old lava flows, one on another: a skyline of shoulders and gullies, not a dome
  float gully=1.-abs(2.*fbm(p/120.+3.,3)-1.);
  float crest=(40.+170.*fbm(vec2(az*6.5,r/900.),4)+70.*(fbm(p/62.,4)-.42)+64.*(gully-.62)+9.*smoothstep(.3,.7,fract(fbm(p/90.+9.,2)*7.)))*mask;
  float cliff=smoothstep(-.3,-.1,x+.25*(fbm(p/170.,3)-.4));
  float back=1.-smoothstep(.3,2.4,x);
  return crest*cliff*back;
}

struct Col{ float h; vec2 id; vec2 c; };
Col column(vec2 p,int i){
  float cs=uS2[i].w;
  vec2 g=p/cs,c=floor(g);
  float best=9.; vec2 bid=c,bctr=c;
  for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++){
    vec2 id=c+vec2(float(x),float(y));
    vec2 ctr=id+.12+.76*hash22(id+uS2[i].z);
    vec2 r=ctr-g; float d=dot(r,r);
    if(d<best){ best=d; bid=id; bctr=ctr; }
  }
  Col k; k.id=bid; k.c=bctr*cs;
  float b=i<NS-1?body(k.c,i):0.;
  if(i==NS-1){ float rc=length(k.c); b=headland(k.c,atan(k.c.x,k.c.y),rc); }
  float jit=(hash21(bid+uS2[i].z*3.1)-.5)*(i<NS-1?.02*uS[i].w:2.6);
  k.h=b>.3?max(b+jit*smoothstep(.0,.2*max(uS[i].w,60.),b),.2):-1.;
  return k;
}

// one view ray: returns radiance (premultiplied by hit), .a=hit
vec4 ray(vec2 pix){
  float az=mix(uRange.x,uRange.y,pix.x),el=mix(uRange.z,uRange.w,pix.y);
  float tanEl=tan(el);
  vec2 dir=vec2(sin(az),cos(az));
  float bestT=1e9; vec4 res=vec4(0.);
  for(int i=0;i<NS;i++){
    float t0,t1,dt;
    if(i<NS-1){
      vec2 c=uS[i].xy;
      float R=uS[i].z*max(uS2[i].x,1.)*1.75;
      float m=dot(c,dir);
      float off=length(c-dir*m);
      if(off>R) continue;
      float hw=sqrt(R*R-off*off);
      t0=m-hw; t1=m+hw;
      dt=uS2[i].w*.2;
      // above everything it could reach
      if(uH+t1*tanEl-t1*t1*uCurv>uS[i].w*1.2&&tanEl>0.&&uH+t0*tanEl-t0*t0*uCurv>uS[i].w*1.2) continue;
    } else {
      if(az<.3||az>1.5) continue;
      t0=1450.; t1=3600.; dt=1.4;
      if(uH+t0*tanEl-t0*t0*uCurv>280.) continue;
    }
    if(t0>=bestT) continue;
    Col prev=column(dir*t0,i);
    for(float t=t0;t<min(t1,bestT);t+=dt){
      vec2 p=dir*t;
      float y=uH+t*tanEl-t*t*uCurv;
      if(y<0.) break;
      Col k=column(p,i);
      if(y<=k.h){
        bestT=t;
        bool wall=any(notEqual(k.id,prev.id));
        vec3 N=wall?normalize(vec3(prev.c.x-k.c.x,.0,prev.c.y-k.c.y)):vec3(0.,1.,0.);
        // weathering: the columns are not one colour and they are cracked across
        float hs=hash21(k.id+1.7);
        float strata=.72+.6*vnoise(vec2(hs*40.,y/(i<NS-1?2.3:9.)))*vnoise(vec2(k.id.x*.7,y/.8));
        float alb=(.026+.05*hs*hs)*strata;
        // light: the horizon behind the rock is the brightest thing in the sky
        vec2 Lh=normalize(uSun.xz);
        float toward=.5+.5*dot(normalize(N.xz+1e-4),Lh);
        float E=mix(.45+.5*toward*toward*1.3,.9,N.y)*uSkyLevel;
        vec3 c=vec3(alb*E)+alb*uSunE*max(dot(N,uSun),0.)*.33;
        // the air between: it goes by range, not by height, so the rock is as dark at the foot as at the top
        float Ta=exp(-t*.001*.115);
        vec3 air=texture(uBg,vec2(az/(2.*PI)+.5,.05)).rgb;
        // cold wet air takes the colour out of what it covers before it lifts it
        vec3 veil=mix(air,vec3(dot(air,vec3(.25,.65,.1)))*vec3(.96,.99,1.04),.75);
        c=c*Ta+veil*(1.-Ta);
        // rock stays dark down to the water: the only white is what the swell throws up against it, a thin
        // broken line a metre or so tall, here and gone along the foot
        float n=vnoise(vec2(az*110.,3.7))*.55+vnoise(vec2(az*340.,9.1))*.3+vnoise(vec2(az*1100.,1.3))*.15;
        float hw=.2+1.9*smoothstep(.42,.8,n);
        float surf=smoothstep(hw,hw*.3,y)*smoothstep(.34,.56,n);
        c=mix(c,air*1.08,surf);
        res=vec4(c,1.);
        break;
      }
      prev=k;
    }
  }
  return res;
}

void main(){
  vec2 px=1./vec2(textureSize(uBg,0))*0.+uPix*vec2(1./(uRange.y-uRange.x),1./(uRange.w-uRange.z));
  vec4 acc=vec4(0.);
  const vec2 SS[4]=vec2[4](vec2(-.375,-.125),vec2(.125,-.375),vec2(.375,.125),vec2(-.125,.375));
  for(int i=0;i<4;i++) acc+=ray(uv+SS[i]*px);
  o=acc*.25;
}
